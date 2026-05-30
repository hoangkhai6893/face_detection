#!/usr/bin/env python3
"""
Face Recognition System
- Uses YOLO for robust face detection (handles distant faces better)
- Uses face_recognition library for identification with error handling
- Optimized for speed and accuracy with configurable parameters
"""

import argparse
import logging
import os
import queue
import sys
import threading
from collections import deque
from typing import List, Tuple, Optional

import cv2
import face_recognition
import numpy as np
import pickle
import time
from ultralytics import YOLO

import config
from core.face_utils import extract_face_region


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        tolerance: float = 0.6,
        detection_confidence: float = 0.3,
        encoding_confidence: float = 0.5,
        padding: int = 15,
        encoding_padding: int = 20,
        max_fps_samples: int = 30,
        encodings_file: Optional[str] = None,
        recognition_interval: int = config.RECOGNITION_INTERVAL,
        yolo_input_width: int = config.YOLO_INPUT_WIDTH,
        motion_absdiff_threshold: float = config.MOTION_ABSDIFF_THRESHOLD,
        motion_mog2_threshold: float = config.MOTION_MOG2_THRESHOLD,
        motion_max_idle_sec: float = config.MOTION_MAX_IDLE_SEC,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the face recognizer.

        Args:
            dataset_path: Path to the root folder containing family member subfolders
            model_path: Path to YOLO face detection model
            tolerance: Face matching tolerance (lower = more strict, default 0.6)
            detection_confidence: Minimum YOLO detection confidence (default 0.3)
            encoding_confidence: Minimum YOLO confidence for encoding (default 0.5)
            padding: Padding around face for recognition
            encoding_padding: Padding around face for creating encoding
            max_fps_samples: Number of samples for FPS calculation
            encodings_file: Custom path for encodings file
            recognition_interval: Re-run face encoding every N frames; reuse cached name otherwise
            yolo_input_width: Resize frame to this width before YOLO (0 = no resize)
            logger: Logger instance
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.tolerance = tolerance
        self.detection_confidence = detection_confidence
        self.encoding_confidence = encoding_confidence
        self.padding = padding
        self.encoding_padding = encoding_padding
        self.recognition_interval = recognition_interval
        self.yolo_input_width = yolo_input_width
        self.motion_absdiff_threshold = motion_absdiff_threshold
        self.motion_mog2_threshold    = motion_mog2_threshold
        self.motion_max_idle_sec      = motion_max_idle_sec
        # MOG2: học background qua 500 frames (~33s ở 15fps), bỏ shadow detection để nhẹ hơn
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.logger = logger or logging.getLogger(__name__)

        self._validate_paths()

        self.logger.info(f"Loading YOLO model from: {model_path}")
        self.yolo_model = YOLO(model_path)

        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self._unique_person_count: int = 0

        self.detection_times = deque(maxlen=max_fps_samples)

        # Per-face recognition cache: list of {bbox, name, frame}
        self._face_cache: List[dict] = []
        self._frame_count: int = 0

        self._load_encodings(encodings_file)

    def _validate_paths(self):
        """Validate required paths exist"""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not os.path.exists(self.dataset_path):
            self.logger.warning(f"Dataset not found: {self.dataset_path}")

    def _get_default_encodings_path(self) -> str:
        """Get default encodings file path (from config.ENCODINGS_DIR)."""
        return os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")

    def _load_encodings(self, encodings_file: Optional[str] = None):
        """Load face encodings from file or create from dataset"""
        save_path = encodings_file or self._get_default_encodings_path()

        if os.path.exists(save_path):
            self.logger.info("Loading existing hybrid encodings...")
            try:
                with open(save_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                self.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                self.logger.warning(f"Error loading encodings: {e}, creating new ones...")
                self._create_encodings_from_dataset(save_path)
        else:
            self._create_encodings_from_dataset(save_path)

        self._unique_person_count = len(set(self.known_face_names))

    def _create_encodings_from_dataset(self, save_path: str):
        """Create face encodings using YOLO detection + face_recognition"""
        self.logger.info("Creating hybrid face encodings from dataset...")

        if not os.path.exists(self.dataset_path):
            self.logger.error(f"Dataset path does not exist: {self.dataset_path}")
            return

        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)

            if not os.path.isdir(person_folder):
                continue

            self.logger.info(f"Processing: {person_name}")

            image_files = [
                f for f in os.listdir(person_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            successful_encodings = 0

            for image_file in image_files:
                image_path = os.path.join(person_folder, image_file)

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    face_encoding = self._extract_face_encoding(image)

                    if face_encoding is not None:
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(person_name)
                        successful_encodings += 1

                except Exception as e:
                    self.logger.debug(f"Error processing {image_file}: {e}")
                    continue

            self.logger.info(f"  Encoded {successful_encodings} images for {person_name}")

        self._save_encodings(save_path)

    def _extract_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face encoding from image using YOLO + face_recognition"""
        results = self.yolo_model(image, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                best_box = result.boxes[best_idx]

                if float(best_box.conf) > self.encoding_confidence:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

                    face_image = extract_face_region(
                        image, x1, y1, x2, y2, self.encoding_padding
                    )

                    if face_image is not None:
                        try:
                            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_face, model="hog")

                            if len(face_locations) > 0:
                                face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
                                if len(face_encodings) > 0:
                                    return face_encodings[0]
                        except Exception:
                            pass

        return self._fallback_extract_encoding(image)

    def _fallback_extract_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: extract encoding directly from full image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            return face_encodings[0] if len(face_encodings) > 0 else None
        except Exception:
            return None

    def _save_encodings(self, save_path: str):
        """Save face encodings to file"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            self.logger.info(f"Saved {len(self.known_face_encodings)} encodings to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving encodings: {e}")

    def detect_faces_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using YOLO

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        try:
            results = self.yolo_model(frame, verbose=False)
            faces = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence > self.detection_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            faces.append((x1, y1, x2, y2, confidence))

            return faces
        except Exception as e:
            self.logger.error(f"YOLO detection error: {e}")
            return []

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Intersection-over-Union for two (x1,y1,x2,y2) boxes"""
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _get_cached_name(self, bbox: Tuple) -> Optional[str]:
        """Return cached name if this face was recognized recently at the same position."""
        for entry in self._face_cache:
            if (self._frame_count - entry['frame'] <= self.recognition_interval and
                    self._iou(bbox, entry['bbox']) > 0.4):
                return entry['name']
        return None

    def _update_cache(self, bbox: Tuple, name: str):
        """Store recognition result for this face position."""
        self._face_cache = [e for e in self._face_cache if self._iou(bbox, e['bbox']) <= 0.4]
        self._face_cache.append({'bbox': bbox, 'name': name, 'frame': self._frame_count})

    def recognize_face_in_region(
        self,
        rgb_frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> str:
        """
        Recognize face in a specific region.

        Args:
            rgb_frame: Full frame in RGB color space (pre-converted for efficiency)
            x1, y1, x2, y2: YOLO bounding box coordinates in rgb_frame space
        """
        try:
            h, w = rgb_frame.shape[:2]

            # Pass the YOLO bbox (with padding) directly as the known face location.
            # This skips dlib's HOG face detection step inside face_encodings(),
            # saving ~30-50% of recognition time with no accuracy loss.
            top    = max(0, y1 - self.padding)
            right  = min(w, x2 + self.padding)
            bottom = min(h, y2 + self.padding)
            left   = max(0, x1 - self.padding)
            face_location = (top, right, bottom, left)  # face_recognition: (top, right, bottom, left)

            face_encodings = face_recognition.face_encodings(
                rgb_frame, known_face_locations=[face_location]
            )

            if len(face_encodings) == 0:
                return "No Face"

            face_encoding = face_encodings[0]

            if len(self.known_face_encodings) > 0:
                distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding
                )
                min_distance = np.min(distances)

                if min_distance <= self.tolerance:
                    best_match_index = np.argmin(distances)
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - min_distance
                    return f"{name} ({confidence:.2f})"
                else:
                    return "Unknown"
            else:
                return "No Training Data"

        except Exception as e:
            self.logger.debug(f"Recognition error: {e}")
            return "Error"

    def draw_results(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, float]],
        names: List[str]
    ) -> np.ndarray:
        """Draw detection results on frame"""
        for (x1, y1, x2, y2, conf), name in zip(detections, names):
            if "Unknown" in name:
                color = (0, 165, 255)
            elif "No Face" in name or "Error" in name:
                color = (0, 0, 255)
            elif "No Training Data" in name:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            det_text = f"Det: {conf:.2f}"
            cv2.putText(frame, det_text, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(frame, (x1, y2), (x1 + 200, y2 + 25), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 5, y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run_recognition(self, camera_id: int = 0, frame_width: int = 1280, frame_height: int = 720):
        """Main recognition loop"""
        self.logger.info("Starting face recognition...")
        self.logger.info(f"  Tolerance: {self.tolerance}")
        self.logger.info(f"  Detection confidence: {self.detection_confidence}")
        self.logger.info(f"  Recognition interval: every {self.recognition_interval} frames")
        self.logger.info(f"  YOLO input width: {self.yolo_input_width}px")
        self.logger.info("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            self.logger.error("Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.logger.info(f"Camera opened: {frame_width}x{frame_height}")

        # --- Capture thread: always keeps the latest frame ready ---
        # This decouples camera I/O from processing so frames don't queue up.
        frame_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_capture = threading.Event()

        def _capture_worker():
            while not stop_capture.is_set():
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put((False, None))
                    return
                # Drop stale frame if main loop hasn't consumed it yet
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put((True, frame))

        capture_thread = threading.Thread(target=_capture_worker, daemon=True)
        capture_thread.start()

        # Pre-compute YOLO scale factor once
        yolo_scale: float = 1.0
        if self.yolo_input_width and self.yolo_input_width < frame_width:
            yolo_scale = self.yolo_input_width / frame_width

        # Motion gate state
        _prev_gray:               Optional[np.ndarray]               = None
        _last_face_detections:    List[Tuple[int, int, int, int, float]] = []
        _last_recognition_results: List[str]                          = []
        _last_yolo_time:          float                               = time.time()

        try:
            while True:
                start_time = time.time()

                ret, frame = frame_queue.get()
                if not ret:
                    self.logger.error("Error reading frame")
                    break

                self._frame_count += 1

                # --- Motion gate (~0.6ms): quyết định có chạy YOLO không ---
                _motion_small = cv2.resize(frame, (160, 90))
                _curr_gray    = cv2.cvtColor(_motion_small, cv2.COLOR_BGR2GRAY)

                # absdiff: kiểm tra nhanh "có gì thay đổi không?" (~0.1ms)
                if _prev_gray is not None:
                    _absdiff_score = float(cv2.absdiff(_prev_gray, _curr_gray).mean())
                else:
                    _absdiff_score = 255.0  # frame đầu tiên: luôn xử lý
                _prev_gray = _curr_gray

                _idle_too_long = (time.time() - _last_yolo_time) > self.motion_max_idle_sec

                if _absdiff_score >= self.motion_absdiff_threshold or _idle_too_long:
                    # MOG2: xác nhận người thật hay chỉ thay đổi ánh sáng (~0.5ms)
                    _mog2_mask  = self._mog2.apply(_curr_gray)
                    _mog2_score = float(_mog2_mask.mean()) / 2.55  # normalize 0–100
                    _should_run = _mog2_score >= self.motion_mog2_threshold or _idle_too_long
                else:
                    _should_run = False

                if _should_run:
                    # --- YOLO detection on a downscaled frame ---
                    if yolo_scale < 1.0:
                        small_h = int(frame_height * yolo_scale)
                        small_frame = cv2.resize(frame, (self.yolo_input_width, small_h))
                        small_detections = self.detect_faces_yolo(small_frame)
                        inv = 1.0 / yolo_scale
                        face_detections = [
                            (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv), c)
                            for x1, y1, x2, y2, c in small_detections
                        ]
                    else:
                        face_detections = self.detect_faces_yolo(frame)

                    # Convert BGR→RGB once per active frame (not once per face)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Prune stale cache entries
                    self._face_cache = [
                        e for e in self._face_cache
                        if self._frame_count - e['frame'] <= self.recognition_interval * 2
                    ]

                    recognition_results = []
                    for x1, y1, x2, y2, conf in face_detections:
                        bbox = (x1, y1, x2, y2)
                        cached = self._get_cached_name(bbox)
                        if cached is not None:
                            # Face was recognized recently — reuse result
                            recognition_results.append(cached)
                        else:
                            # New face or cache expired — run full recognition
                            name = self.recognize_face_in_region(rgb_frame, x1, y1, x2, y2)
                            self._update_cache(bbox, name)
                            recognition_results.append(name)

                    _last_yolo_time          = time.time()
                    _last_face_detections    = face_detections
                    _last_recognition_results = recognition_results
                else:
                    # Static scene: tái dùng toàn bộ kết quả cũ — không chạy YOLO, BGR→RGB, hay dlib
                    face_detections    = _last_face_detections
                    recognition_results = _last_recognition_results

                if face_detections and recognition_results:
                    frame = self.draw_results(frame, face_detections, recognition_results)

                detection_time = time.time() - start_time
                self.detection_times.append(detection_time)

                avg_time = np.mean(self.detection_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0

                _indicator = "M" if _should_run else "-"
                info_text = [
                    f"FPS: {fps:.1f} [{_indicator}]",
                    f"Faces: {len(face_detections)}",
                    f"Known: {self._unique_person_count} people",
                    f"Encodings: {len(self.known_face_encodings)}"
                ]

                for i, text in enumerate(info_text):
                    y_pos = 30 + i * 25
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Face Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            self.logger.info("Stopped by user")

        finally:
            stop_capture.set()
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Recognition system stopped")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Face Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=config.DATASET_PATH,
        help='Path to dataset folder containing person subfolders'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=config.MODEL_PATH,
        help='Path to YOLO face detection model'
    )

    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=config.TOLERANCE,
        help='Face matching tolerance (lower = more strict)'
    )

    parser.add_argument(
        '--detection-confidence',
        type=float,
        default=config.DETECTION_CONFIDENCE,
        help='Minimum YOLO detection confidence'
    )

    parser.add_argument(
        '--encoding-confidence',
        type=float,
        default=config.ENCODING_CONFIDENCE,
        help='Minimum YOLO confidence for encoding'
    )

    parser.add_argument(
        '--padding', '-p',
        type=int,
        default=config.RECOGNITION_PADDING,
        help='Padding around face for recognition'
    )

    parser.add_argument(
        '--encodings-file',
        type=str,
        default=None,
        help='Custom path for encodings file'
    )

    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=config.FRAME_WIDTH,
        help='Camera frame width'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=config.FRAME_HEIGHT,
        help='Camera frame height'
    )

    parser.add_argument(
        '--recognition-interval',
        type=int,
        default=config.RECOGNITION_INTERVAL,
        help='Re-run face encoding every N frames; higher = faster but slower name updates'
    )

    parser.add_argument(
        '--yolo-input-width',
        type=int,
        default=config.YOLO_INPUT_WIDTH,
        help='Resize frame to this width before YOLO (0 = no resize)'
    )

    parser.add_argument(
        '--motion-absdiff',
        type=float,
        default=config.MOTION_ABSDIFF_THRESHOLD,
        help='Ngưỡng absdiff phát hiện chuyển động (thấp=nhạy hơn, 0=tắt gate)'
    )

    parser.add_argument(
        '--motion-mog2',
        type=float,
        default=config.MOTION_MOG2_THRESHOLD,
        help='Ngưỡng %% pixels MOG2 xác nhận motion thật (thấp=nhạy hơn)'
    )

    parser.add_argument(
        '--motion-idle',
        type=float,
        default=config.MOTION_MAX_IDLE_SEC,
        help='Giây tối đa không chạy YOLO trước khi force-run (fallback người đứng yên)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    logger = setup_logging(args.verbose)

    try:
        recognizer = FaceRecognizer(
            dataset_path=args.dataset,
            model_path=args.model,
            tolerance=args.tolerance,
            detection_confidence=args.detection_confidence,
            encoding_confidence=args.encoding_confidence,
            padding=args.padding,
            encodings_file=args.encodings_file,
            recognition_interval=args.recognition_interval,
            yolo_input_width=args.yolo_input_width,
            motion_absdiff_threshold=args.motion_absdiff,
            motion_mog2_threshold=args.motion_mog2,
            motion_max_idle_sec=args.motion_idle,
            logger=logger
        )

        recognizer.run_recognition(
            camera_id=args.camera,
            frame_width=args.width,
            frame_height=args.height
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
