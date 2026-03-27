#!/usr/bin/env python3
"""
Improved Hybrid Face Recognition System
- Uses YOLO for robust face detection (handles distant faces better)
- Uses face_recognition for identification with error handling
- Optimized for speed and accuracy with configurable parameters
"""

import argparse
import logging
import os
import sys
from collections import deque
from typing import List, Tuple, Optional

import cv2
import face_recognition
import numpy as np
import pickle
import time
from ultralytics import YOLO


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


class ImprovedHybridFaceRecognizer:
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
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the improved hybrid face recognizer
        
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
            logger: Logger instance
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.tolerance = tolerance
        self.detection_confidence = detection_confidence
        self.encoding_confidence = encoding_confidence
        self.padding = padding
        self.encoding_padding = encoding_padding
        self.logger = logger or logging.getLogger(__name__)
        
        self._validate_paths()
        
        self.logger.info(f"Loading YOLO model from: {model_path}")
        self.yolo_model = YOLO(model_path)
        
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        
        self.detection_times = deque(maxlen=max_fps_samples)
        
        self._load_encodings(encodings_file)
    
    def _validate_paths(self):
        """Validate required paths exist"""
        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if not os.path.exists(self.dataset_path):
            self.logger.warning(f"Dataset not found: {self.dataset_path}")
    
    def _get_default_encodings_path(self) -> str:
        """Get default encodings file path"""
        return os.path.join(
            os.path.dirname(self.dataset_path), 
            "face_encodings_hybrid.pkl"
        )
    
    def _pad_box(
        self, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int, 
        frame_shape: Tuple[int, ...], 
        padding: int
    ) -> Tuple[int, int, int, int]:
        """Apply padding to bounding box with boundary checking"""
        height, width = frame_shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(width, x2 + padding)
        y2_pad = min(height, y2 + padding)
        return x1_pad, y1_pad, x2_pad, y2_pad
    
    def _extract_face_region(
        self, 
        frame: np.ndarray, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int, 
        padding: int
    ) -> Optional[np.ndarray]:
        """Extract face region with padding"""
        x1_pad, y1_pad, x2_pad, y2_pad = self._pad_box(
            x1, y1, x2, y2, frame.shape, padding
        )
        face_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        return face_region if face_region.size > 0 else None
    
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
                    
                    face_image = self._extract_face_region(
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
    
    def recognize_face_in_region(
        self, 
        frame: np.ndarray, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int
    ) -> str:
        """Recognize face in a specific region"""
        try:
            face_region = self._extract_face_region(frame, x1, y1, x2, y2, self.padding)
            
            if face_region is None:
                return "Invalid Region"
            
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)
            
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
            return f"Error"
    
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
        self.logger.info("Starting improved hybrid face recognition...")
        self.logger.info(f"  Tolerance: {self.tolerance}")
        self.logger.info(f"  Detection confidence: {self.detection_confidence}")
        self.logger.info("Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error("Cannot open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        self.logger.info(f"Camera opened: {frame_width}x{frame_height}")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Error reading frame")
                    break
                
                face_detections = self.detect_faces_yolo(frame)
                
                recognition_results = []
                for x1, y1, x2, y2, conf in face_detections:
                    name = self.recognize_face_in_region(frame, x1, y1, x2, y2)
                    recognition_results.append(name)
                
                if face_detections and recognition_results:
                    frame = self.draw_results(frame, face_detections, recognition_results)
                
                detection_time = time.time() - start_time
                self.detection_times.append(detection_time)
                
                avg_time = np.mean(self.detection_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                info_text = [
                    f"FPS: {fps:.1f}",
                    f"Faces: {len(face_detections)}",
                    f"Known: {len(set(self.known_face_names))} people",
                    f"Encodings: {len(self.known_face_encodings)}"
                ]
                
                for i, text in enumerate(info_text):
                    y_pos = 30 + i * 25
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Improved Hybrid Face Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Recognition system stopped")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Improved Hybrid Face Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='/home/dkhai/workspace/family_images',
        help='Path to dataset folder containing person subfolders'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/home/dkhai/workspace/src/yolov11n-face.pt',
        help='Path to YOLO face detection model'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=0.6,
        help='Face matching tolerance (lower = more strict)'
    )
    
    parser.add_argument(
        '--detection-confidence',
        type=float,
        default=0.3,
        help='Minimum YOLO detection confidence'
    )
    
    parser.add_argument(
        '--encoding-confidence',
        type=float,
        default=0.5,
        help='Minimum YOLO confidence for encoding'
    )
    
    parser.add_argument(
        '--padding', '-p',
        type=int,
        default=15,
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
        default=1280,
        help='Camera frame width'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera frame height'
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
        recognizer = ImprovedHybridFaceRecognizer(
            dataset_path=args.dataset,
            model_path=args.model,
            tolerance=args.tolerance,
            detection_confidence=args.detection_confidence,
            encoding_confidence=args.encoding_confidence,
            padding=args.padding,
            encodings_file=args.encodings_file,
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
