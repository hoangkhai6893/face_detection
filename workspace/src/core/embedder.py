#!/usr/bin/env python3
"""
FaceEmbedder — abstract interface for face embedding & matching.

Distance convention (IMPORTANT — both backends use same semantics):
  0.0 = completely identical
  higher = more different
  → TOLERANCE and CONFUSION_MARGIN in config work the same for both backends.

DlibEmbedder:  Euclidean distance via face_recognition.face_distance()
SFaceEmbedder: cosine distance = 1 - cosine_score  (normalized to [0, 1])
               threshold reference: 1 - 0.363 ≈ 0.637 for same-identity
"""

from __future__ import annotations

import abc
from typing import List, Optional

import cv2
import numpy as np

from core.detector import FaceDetection


class FaceEmbedder(abc.ABC):

    @abc.abstractmethod
    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 15,
    ) -> Optional[np.ndarray]:
        """
        Extract embedding vector from a detected face.

        Args:
            frame_bgr: Full BGR frame (NOT a cropped region)
            detection: FaceDetection with bbox (and landmarks for SFace)
            padding: Extra pixels around bbox (used by DlibEmbedder; SFace uses alignCrop)
        Returns:
            1-D float32 embedding vector, or None if extraction fails.
        """
        ...

    @abc.abstractmethod
    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distances between query and every known encoding.

        Returns:
            np.ndarray shape (N,) — lower = more similar.
        """
        ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# DlibEmbedder — wraps face_recognition (dlib ResNet 128-d)
# ---------------------------------------------------------------------------

class DlibEmbedder(FaceEmbedder):
    """Wrapper around face_recognition (dlib ResNet). Preserves logic from recognizer.py.

    encode() passes the YOLO bbox + padding directly as known_face_locations,
    skipping dlib's internal HOG step (same optimization as recognizer.py).
    """

    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 15,
    ) -> Optional[np.ndarray]:
        import face_recognition
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        top    = max(0, y1 - padding)
        right  = min(w, x2 + padding)
        bottom = min(h, y2 + padding)
        left   = max(0, x1 - padding)
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(
                rgb, known_face_locations=[(top, right, bottom, left)]
            )
            return encodings[0] if encodings else None
        except Exception:
            return None

    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        import face_recognition
        return face_recognition.face_distance(known_encodings, query_encoding)


# ---------------------------------------------------------------------------
# SFaceEmbedder — cv2.FaceRecognizerSF (Phase 3 target)
# ---------------------------------------------------------------------------

class SFaceEmbedder(FaceEmbedder):
    """Face embedder using SFace (cv2.FaceRecognizerSF) with alignCrop.

    MUST be paired with YuNetDetector — requires FaceDetection._raw_row
    (the landmark-annotated YuNet output row) for alignCrop().

    encode() → 128-d L2-normalized feature vector.
    batch_distance() → cosine distance = 1 - cosine_score ∈ [0, 1].
      0 = same person, 1 = completely different.

    Official threshold: cosine_score >= 0.363 → same person
      → distance threshold ≈ 0.637 (retune in Phase 4 for your family data).
    """

    # FR_COSINE = 1 in all OpenCV builds that support SFace.
    # Resolved at class-definition time via getattr to avoid AttributeError on
    # older builds where the constant is not yet exposed as a Python attribute.
    _COSINE_MODE: int = getattr(
        getattr(cv2, "FaceRecognizerSF", None), "FR_COSINE", 1
    )

    def __init__(self, model_path: str) -> None:
        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")

    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 0,   # SFace uses alignCrop internally, no manual padding
    ) -> Optional[np.ndarray]:
        """Align-crop the face using landmarks, then extract SFace feature."""
        if detection._raw_row is None:
            return None   # YuNet detection required (has _raw_row with landmarks)
        try:
            aligned = self._recognizer.alignCrop(frame_bgr, detection._raw_row)
            feature = self._recognizer.feature(aligned)   # shape (1, 128)
            return feature.flatten().astype(np.float32)   # shape (128,)
        except Exception:
            return None

    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine distance = 1 - cosine_score for each known encoding."""
        distances = np.empty(len(known_encodings), dtype=np.float32)
        q = query_encoding.reshape(1, -1).astype(np.float32)
        for i, known in enumerate(known_encodings):
            k = known.reshape(1, -1).astype(np.float32)
            score = self._recognizer.match(k, q, self._COSINE_MODE)
            distances[i] = max(0.0, 1.0 - float(score))   # clamp to [0, 1]
        return distances
