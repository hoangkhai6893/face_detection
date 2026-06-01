#!/usr/bin/env python3
"""
FaceDetector — abstract interface for face detection.

Backends:
  YoloDetector  — ultralytics YOLO (current stack, default)
  YuNetDetector — cv2.FaceDetectorYN (Phase 3 target, lightweight OpenCV-native)

Distance convention: both backends return FaceDetection with (x1,y1,x2,y2) bbox
in the original frame's coordinate space. YuNetDetector also provides 5 landmarks
and a raw_row for SFaceEmbedder.alignCrop().
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FaceDetection:
    """Immutable result for a single detected face."""
    __slots__ = ("bbox", "confidence", "landmarks", "_raw_row")

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],           # (x1, y1, x2, y2) in original frame
        confidence: float,
        landmarks: Optional[np.ndarray] = None,    # shape (5, 2) [x, y] per point, or None
        _raw_row: Optional[np.ndarray] = None,     # raw YuNet output row for alignCrop
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        self._raw_row = _raw_row


class FaceDetector(abc.ABC):
    """Interface: receives BGR frame, returns List[FaceDetection]."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Args:
            frame: BGR image (numpy array, uint8)
        Returns:
            List of FaceDetection filtered by detection_confidence threshold.
        """
        ...

    def warmup(self, size: Tuple[int, int] = (64, 64)) -> None:
        """Run one dummy inference to trigger JIT/ONNX compilation. Call after __init__."""
        dummy = np.zeros((*size, 3), dtype=np.uint8)
        self.detect(dummy)

    def close(self) -> None:
        """Release resources if needed."""
        pass


# ---------------------------------------------------------------------------
# YoloDetector — wraps ultralytics YOLO (current stack)
# ---------------------------------------------------------------------------

class YoloDetector(FaceDetector):
    """Wrapper around ultralytics YOLO. Preserves logic from recognizer.py.

    Handles frame downscaling to input_width internally so callers always
    receive bboxes in the original frame coordinate space.
    """

    def __init__(
        self,
        model_path: str,
        detection_confidence: float = 0.3,
        input_width: int = 416,
    ) -> None:
        from ultralytics import YOLO as _YOLO
        self._model = _YOLO(model_path)
        self._conf = detection_confidence
        self._input_width = input_width
        self.warmup()

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]

        if self._input_width and self._input_width < w:
            scale = self._input_width / w
            small = cv2.resize(frame, (self._input_width, int(h * scale)))
            inv = 1.0 / scale
        else:
            small, inv = frame, 1.0

        results = self._model(small, verbose=False)
        detections: List[FaceDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bbox = (
                    max(0, int(x1 * inv)),
                    max(0, int(y1 * inv)),
                    min(w, int(x2 * inv)),
                    min(h, int(y2 * inv)),
                )
                detections.append(FaceDetection(bbox=bbox, confidence=conf))

        return detections


# ---------------------------------------------------------------------------
# YuNetDetector — cv2.FaceDetectorYN (Phase 3 target)
# ---------------------------------------------------------------------------

class YuNetDetector(FaceDetector):
    """Face detector using YuNet (cv2.FaceDetectorYN).

    Returns 5 facial landmarks alongside each bbox so SFaceEmbedder can
    call alignCrop() for accurate embedding.

    YuNet output format (N×15 matrix):
      col 0-3:  x, y, w, h  (top-left + size — NOT x1,y1,x2,y2)
      col 4-13: 5 landmarks (x,y) × 5
      col 14:   confidence score
    """

    def __init__(
        self,
        model_path: str,
        detection_confidence: float = 0.6,
        input_size: Tuple[int, int] = (320, 240),  # (width, height)
        nms_threshold: float = 0.3,
    ) -> None:
        self._conf = detection_confidence
        self._input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=detection_confidence,
            nms_threshold=nms_threshold,
            top_k=5,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        orig_h, orig_w = frame.shape[:2]
        iw, ih = self._input_size

        resized = cv2.resize(frame, (iw, ih))
        self._detector.setInputSize((iw, ih))

        _, faces = self._detector.detect(resized)
        if faces is None:
            return []

        sx = orig_w / iw
        sy = orig_h / ih

        detections: List[FaceDetection] = []
        for row in faces:
            conf = float(row[14])
            if conf < self._conf:
                continue

            # Convert YuNet (x,y,w,h) → (x1,y1,x2,y2) in original frame space
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1 = max(0, int(x * sx))
            y1 = max(0, int(y * sy))
            x2 = min(orig_w, int((x + w) * sx))
            y2 = min(orig_h, int((y + h) * sy))

            # Scale landmarks to original frame space
            landmarks = np.array(
                [[row[4 + i * 2] * sx, row[4 + i * 2 + 1] * sy] for i in range(5)],
                dtype=np.float32,
            )  # shape (5, 2)

            # Build raw_row scaled to original frame (for alignCrop)
            raw_row = row.copy()
            raw_row[0] = x * sx
            raw_row[1] = y * sy
            raw_row[2] = w * sx
            raw_row[3] = h * sy
            for i in range(5):
                raw_row[4 + i * 2]     = row[4 + i * 2] * sx
                raw_row[4 + i * 2 + 1] = row[4 + i * 2 + 1] * sy

            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                landmarks=landmarks,
                _raw_row=raw_row,
            ))

        return detections
