#!/usr/bin/env python3
"""
Face Recognition System — pluggable detector/embedder backends.

Backends (FACE_BACKEND env or --face-backend arg):
  "dlib"  — YOLO + face_recognition (default, current stack)
  "sface" — YuNet + SFace (Phase 3, OpenCV-native, lighter on ARM)

Phase 1 features:
  --headless / HEADLESS=true   → no cv2.imshow (required for Pi headless)
  ACTIVE_PROCESS_EVERY         → skip frames when ACTIVE (reduce CPU)
  HOG fallback WARNING + count → visibility into slow fallback path
  Health log thread            → hourly FPS / state / face count report
"""

import argparse
import logging
import os
import pickle
import queue
import sys
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

import config
from core.detector import FaceDetection, FaceDetector
from core.embedder import FaceEmbedder
from core.motion_guard import MotionGuard


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def _build_backends(
    backend: str,
    model_path: str,
    detection_confidence: float,
    yolo_input_width: int,
) -> Tuple[FaceDetector, FaceEmbedder]:
    """Factory: instantiate (detector, embedder) pair for the requested backend."""
    if backend == "sface":
        from core.detector import YuNetDetector
        from core.embedder import SFaceEmbedder
        detector: FaceDetector = YuNetDetector(
            model_path=config.YUNET_MODEL_PATH,
            detection_confidence=detection_confidence,
        )
        embedder: FaceEmbedder = SFaceEmbedder(model_path=config.SFACE_MODEL_PATH)
    else:
        from core.detector import YoloDetector
        from core.embedder import DlibEmbedder
        detector = YoloDetector(
            model_path=model_path,
            detection_confidence=detection_confidence,
            input_width=yolo_input_width,
        )
        embedder = DlibEmbedder()
    return detector, embedder


class FaceRecognizer:
    def __init__(
        self,
        dataset_path: str,
        model_path: str,
        tolerance: float = config.TOLERANCE,
        detection_confidence: float = config.DETECTION_CONFIDENCE,
        encoding_confidence: float = config.ENCODING_CONFIDENCE,
        padding: int = config.RECOGNITION_PADDING,
        encoding_padding: int = config.ENCODING_PADDING,
        max_fps_samples: int = config.MAX_FPS_SAMPLES,
        encodings_file: Optional[str] = None,
        recognition_interval: int = config.RECOGNITION_INTERVAL,
        yolo_input_width: int = config.YOLO_INPUT_WIDTH,
        motion_absdiff_threshold: float = config.MOTION_ABSDIFF_THRESHOLD,
        motion_mog2_threshold: float = config.MOTION_MOG2_THRESHOLD,
        motion_max_idle_sec: float = config.MOTION_MAX_IDLE_SEC,
        recognition_top_k: int = config.RECOGNITION_TOP_K,
        iou_cache_threshold: float = config.IOU_CACHE_THRESHOLD,
        confusion_margin: float = config.CONFUSION_MARGIN,
        headless: bool = config.HEADLESS,
        active_process_every: int = config.ACTIVE_PROCESS_EVERY,
        face_backend: str = config.FACE_BACKEND,
        logger: Optional[logging.Logger] = None,
        on_recognition_update: Optional[Callable[[str, tuple, "np.ndarray"], None]] = None,
    ):
        self.dataset_path       = dataset_path
        self.model_path         = model_path
        self.tolerance          = tolerance
        self.detection_confidence = detection_confidence
        self.encoding_confidence  = encoding_confidence
        self.padding            = padding
        self.encoding_padding   = encoding_padding
        self.recognition_interval = recognition_interval
        self.recognition_top_k  = recognition_top_k
        self.iou_cache_threshold = iou_cache_threshold
        self.confusion_margin   = confusion_margin
        self.headless           = headless
        self.active_process_every = max(1, active_process_every)
        self.face_backend       = face_backend
        self.logger             = logger or logging.getLogger(__name__)
        self.on_recognition_update = on_recognition_update

        # Phase 1: performance counters
        self._hog_fallback_count  = 0
        self._active_skip_counter = 0
        self._health_faces_seen   = 0
        self._health_start        = time.time()

        self._motion_guard = MotionGuard(
            absdiff_threshold=motion_absdiff_threshold,
            mog2_threshold=motion_mog2_threshold,
            max_idle_sec=motion_max_idle_sec,
            logger=self.logger,
        )

        self._validate_paths()

        # Phase 2+3: pluggable backends
        self.logger.info("Loading backend: %s (model: %s)", face_backend, model_path)
        self.detector, self.embedder = _build_backends(
            face_backend, model_path, detection_confidence, yolo_input_width
        )

        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names:     List[str]         = []
        self._unique_person_count: int               = 0
        self.detection_times = deque(maxlen=max_fps_samples)
        self._face_cache: List[dict] = []
        self._frame_count: int = 0

        self._load_encodings(encodings_file)
        self._start_health_log_thread()

    # ------------------------------------------------------------------
    # Paths & encodings
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        if not os.path.exists(self.model_path) and self.face_backend == "dlib":
            self.logger.error("Model not found: %s", self.model_path)
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.dataset_path):
            self.logger.warning("Dataset not found: %s", self.dataset_path)

    def _get_default_encodings_path(self) -> str:
        return os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")

    def _load_encodings(self, encodings_file: Optional[str] = None) -> None:
        save_path = encodings_file or self._get_default_encodings_path()
        if os.path.exists(save_path):
            self.logger.info("Loading encodings from %s", save_path)
            try:
                with open(save_path, "rb") as f:
                    data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names     = data["names"]
                saved_backend = data.get("backend", "dlib")
                if saved_backend != self.face_backend:
                    self.logger.warning(
                        "Encodings were built with backend='%s' but current backend='%s'. "
                        "Run rebuild_encodings to regenerate.",
                        saved_backend, self.face_backend,
                    )
                self.logger.info("Loaded %d encodings", len(self.known_face_encodings))
            except Exception as e:
                self.logger.warning("Error loading encodings: %s — rebuilding", e)
                self._create_encodings_from_dataset(save_path)
        else:
            self._create_encodings_from_dataset(save_path)
        self._unique_person_count = len(set(self.known_face_names))

    def _create_encodings_from_dataset(self, save_path: str) -> None:
        """Fallback: build .pkl on-the-fly from dataset. Prefer rebuild_encodings() instead."""
        self.logger.info("Building encodings from dataset (fallback)…")
        if not os.path.exists(self.dataset_path):
            self.logger.error("Dataset path missing: %s", self.dataset_path)
            return

        for person_name in os.listdir(self.dataset_path):
            folder = os.path.join(self.dataset_path, person_name)
            if not os.path.isdir(folder):
                continue
            self.logger.info("Processing: %s", person_name)
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                image = cv2.imread(os.path.join(folder, fname))
                if image is None:
                    continue
                enc = self._extract_face_encoding(image)
                if enc is not None:
                    self.known_face_encodings.append(enc)
                    self.known_face_names.append(person_name)

        self._save_encodings(save_path)

    def _extract_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract encoding using detector + embedder (training-time path)."""
        detections = self.detector.detect(image)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            enc = self.embedder.encode(image, best, padding=self.encoding_padding)
            if enc is not None:
                return enc
        return self._fallback_extract_encoding(image)

    def _fallback_extract_encoding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """HOG full-image fallback when detector misses. Logs WARNING + timing."""
        t0 = time.time()
        try:
            import face_recognition as _fr
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            encs = _fr.face_encodings(rgb)
            elapsed_ms = (time.time() - t0) * 1000
            if encs:
                self._hog_fallback_count += 1
                self.logger.warning(
                    "HOG fallback triggered (%.0fms) — total: %d. "
                    "Consider improving dataset image quality.",
                    elapsed_ms, self._hog_fallback_count,
                )
                return encs[0]
        except ImportError:
            pass   # face_recognition removed in Phase 3 — graceful no-op
        except Exception:
            pass
        return None

    def _save_encodings(self, save_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({
                    "encodings": self.known_face_encodings,
                    "names":     self.known_face_names,
                    "backend":   self.face_backend,
                }, f)
            self.logger.info("Saved %d encodings → %s", len(self.known_face_encodings), save_path)
        except Exception as e:
            self.logger.error("Error saving encodings: %s", e)

    # ------------------------------------------------------------------
    # IOU cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _get_cached_name(self, bbox: Tuple) -> Optional[str]:
        for entry in self._face_cache:
            if (self._frame_count - entry["frame"] <= self.recognition_interval and
                    self._iou(bbox, entry["bbox"]) > self.iou_cache_threshold):
                return entry["name"]
        return None

    def _update_cache(self, bbox: Tuple, name: str) -> None:
        self._face_cache = [
            e for e in self._face_cache
            if self._iou(bbox, e["bbox"]) <= self.iou_cache_threshold
        ]
        self._face_cache.append({"bbox": bbox, "name": name, "frame": self._frame_count})

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize_face_in_region(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
    ) -> str:
        """Recognize a detected face using the current embedder.

        Args:
            frame_bgr: Full BGR frame (not a crop)
            detection: FaceDetection with bbox (and _raw_row for SFace)
        """
        face_encoding = self.embedder.encode(frame_bgr, detection, padding=self.padding)
        if face_encoding is None:
            return "No Face"

        if not self.known_face_encodings:
            return "No Training Data"

        distances = self.embedder.batch_distance(self.known_face_encodings, face_encoding)

        # Top-K weighted voting: weight = (1 - distance) → closer encodings vote harder
        k = min(self.recognition_top_k, len(distances))
        top_k_idx = np.argsort(distances)[:k]

        votes: dict = {}
        for idx in top_k_idx:
            if distances[idx] <= self.tolerance:
                person = self.known_face_names[idx]
                votes[person] = votes.get(person, 0.0) + (1.0 - distances[idx])

        if not votes:
            return "Unknown"

        winner = max(votes, key=lambda p: votes[p])
        best_dist = min(distances[i] for i in top_k_idx if self.known_face_names[i] == winner)

        # Margin guard: reject if runner-up is too close (prevents family member confusion)
        if len(set(self.known_face_names)) > 1:
            second_best_dist = min(
                (distances[i] for i, n in enumerate(self.known_face_names) if n != winner),
                default=float("inf"),
            )
            if second_best_dist - best_dist < self.confusion_margin:
                return "Unknown"

        return f"{winner} ({1.0 - best_dist:.2f})"

    def detect_faces_yolo(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """Compatibility shim — delegates to self.detector.detect()."""
        return [
            (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence)
            for d in self.detector.detect(frame)
        ]

    def draw_results(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int, float]],
        names: List[str],
    ) -> np.ndarray:
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
            cv2.putText(frame, f"Det: {conf:.2f}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + 200, y2 + 25), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 5, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    # ------------------------------------------------------------------
    # Health log (Phase 1)
    # ------------------------------------------------------------------

    def _start_health_log_thread(self) -> None:
        def _worker():
            while True:
                time.sleep(3600)
                avg_time = float(np.mean(self.detection_times)) if self.detection_times else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.logger.info(
                    "[HEALTH] uptime=%.1fh | fps=%.1f | state=%s | "
                    "faces_seen=%d | hog_fallback=%d | backend=%s",
                    (time.time() - self._health_start) / 3600,
                    fps,
                    self._motion_guard.state.name,
                    self._health_faces_seen,
                    self._hog_fallback_count,
                    self.face_backend,
                )
                self._health_faces_seen = 0  # reset per-hour counter

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Main recognition loop
    # ------------------------------------------------------------------

    def run_recognition(
        self,
        camera_id: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> None:
        self.logger.info("Starting face recognition…")
        self.logger.info("  Backend: %s | Headless: %s | ACTIVE_PROCESS_EVERY: %d",
                         self.face_backend, self.headless, self.active_process_every)
        self.logger.info("  Tolerance: %.2f | Recognition interval: %d frames",
                         self.tolerance, self.recognition_interval)
        if not self.headless:
            self.logger.info("  Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error("Cannot open camera %d", camera_id)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.logger.info("Camera opened: %dx%d", frame_width, frame_height)

        # Capture thread: decouple camera I/O from processing
        frame_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_capture = threading.Event()

        def _capture_worker():
            while not stop_capture.is_set():
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put((False, None))
                    return
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put((True, frame))

        capture_thread = threading.Thread(target=_capture_worker, daemon=True)
        capture_thread.start()

        _last_face_detections:     List[Tuple[int, int, int, int, float]] = []
        _last_recognition_results: List[str]                               = []
        _active_skip = 0   # ACTIVE_PROCESS_EVERY counter (local to loop)

        try:
            while True:
                start_time = time.time()

                ret, frame = frame_queue.get()
                if not ret:
                    self.logger.error("Camera read error — stopping")
                    break

                self._frame_count += 1

                # --- State machine (IDLE / ACTIVE) ---
                _should_run = self._motion_guard.should_process(frame)

                # Phase 1: ACTIVE throttle — skip frames to reduce CPU when ACTIVE
                if _should_run and self.active_process_every > 1 and self._motion_guard.is_active:
                    _active_skip = (_active_skip + 1) % self.active_process_every
                    if _active_skip != 0:
                        _should_run = False

                if _should_run:
                    # --- Detection (detector handles resize/scale internally) ---
                    raw_detections = self.detector.detect(frame)
                    face_detections = [
                        (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence)
                        for d in raw_detections
                    ]

                    self._motion_guard.report_faces(len(face_detections))

                    # Prune stale cache entries
                    self._face_cache = [
                        e for e in self._face_cache
                        if self._frame_count - e["frame"] <= self.recognition_interval * 2
                    ]

                    recognition_results: List[str] = []
                    for i, (x1, y1, x2, y2, conf) in enumerate(face_detections):
                        bbox = (x1, y1, x2, y2)
                        cached = self._get_cached_name(bbox)
                        if cached is not None:
                            recognition_results.append(cached)
                        else:
                            name = self.recognize_face_in_region(frame, raw_detections[i])
                            self._update_cache(bbox, name)
                            recognition_results.append(name)
                            if self.on_recognition_update is not None:
                                try:
                                    self.on_recognition_update(name, bbox, frame.copy())
                                except Exception as _e:
                                    self.logger.debug("on_recognition_update error: %s", _e)

                    _last_face_detections     = face_detections
                    _last_recognition_results = recognition_results

                    if face_detections:
                        self._health_faces_seen += len(face_detections)
                else:
                    face_detections     = _last_face_detections
                    recognition_results = _last_recognition_results

                # --- Display (headless guard) ---
                detection_time = time.time() - start_time
                self.detection_times.append(detection_time)

                if not self.headless:
                    if face_detections and recognition_results:
                        frame = self.draw_results(frame, face_detections, recognition_results)

                    avg_time = float(np.mean(self.detection_times))
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    _state  = self._motion_guard.state.name
                    _ind    = "A" if _should_run else "I"
                    for i, text in enumerate([
                        f"FPS: {fps:.1f} [{_state}|{_ind}]",
                        f"Faces: {len(face_detections)}",
                        f"Known: {self._unique_person_count} people",
                        f"Backend: {self.face_backend}",
                    ]):
                        cv2.putText(frame, text, (10, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Face Recognition", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        finally:
            stop_capture.set()
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.logger.info("Recognition stopped. HOG fallbacks: %d", self._hog_fallback_count)


# ---------------------------------------------------------------------------
# Standalone entry point (python src/core/recognizer.py)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default=config.DATASET_PATH)
    parser.add_argument("--model", "-m", default=config.MODEL_PATH)
    parser.add_argument("--tolerance", "-t", type=float, default=config.TOLERANCE)
    parser.add_argument("--detection-confidence", type=float, default=config.DETECTION_CONFIDENCE)
    parser.add_argument("--encoding-confidence", type=float, default=config.ENCODING_CONFIDENCE)
    parser.add_argument("--padding", "-p", type=int, default=config.RECOGNITION_PADDING)
    parser.add_argument("--encodings-file", default=None)
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    parser.add_argument("--recognition-interval", type=int, default=config.RECOGNITION_INTERVAL)
    parser.add_argument("--yolo-input-width", type=int, default=config.YOLO_INPUT_WIDTH)
    parser.add_argument("--motion-absdiff", type=float, default=config.MOTION_ABSDIFF_THRESHOLD)
    parser.add_argument("--motion-mog2", type=float, default=config.MOTION_MOG2_THRESHOLD)
    parser.add_argument("--motion-idle", type=float, default=config.MOTION_MAX_IDLE_SEC)
    parser.add_argument("--headless", action="store_true", default=config.HEADLESS)
    parser.add_argument("--active-process-every", type=int, default=config.ACTIVE_PROCESS_EVERY)
    parser.add_argument("--face-backend", default=config.FACE_BACKEND,
                        choices=["dlib", "sface"], help="Detection+embedding backend")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
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
            headless=args.headless,
            active_process_every=args.active_process_every,
            face_backend=args.face_backend,
            logger=logger,
        )
        recognizer.run_recognition(
            camera_id=args.camera,
            frame_width=args.width,
            frame_height=args.height,
        )
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
