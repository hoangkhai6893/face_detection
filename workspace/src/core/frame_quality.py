from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config


class FrameQualityChecker:
    """Evaluates individual frame quality before accepting it into the dataset."""

    def __init__(
        self,
        min_laplacian: float = config.FRAME_MIN_LAPLACIAN,
        min_brightness: float = 0.10,
        max_brightness: float = 0.92,
        min_face_size: int = config.FRAME_MIN_FACE_PX,
    ):
        self.min_laplacian = min_laplacian
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def is_sharp(self, face_crop: np.ndarray) -> bool:
        """Return True if the face crop is sharp enough (Laplacian variance)."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= self.min_laplacian

    def is_bright(self, face_crop: np.ndarray) -> bool:
        """Return True if brightness is within the acceptable range.

        Mirrors the fix in face_learning_mode.py: accept a wide band
        (0.10 – 0.92) so that dark-room shots still pass; only extreme
        under/over-exposure is rejected.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        return self.min_brightness <= brightness <= self.max_brightness

    def is_large_enough(self, face_crop: np.ndarray) -> bool:
        """Return True if the face crop meets the minimum size requirement."""
        h, w = face_crop.shape[:2]
        return min(h, w) >= self.min_face_size

    # ------------------------------------------------------------------
    # Combined score
    # ------------------------------------------------------------------

    def score(self, face_crop: np.ndarray) -> float:
        """Compute a combined quality score in [0, 1].

        Weights:
          sharpness  40 %
          brightness 30 %  (penalises extremes, plateaus in the good range)
          face size  30 %
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Sharpness
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, lap_var / 500.0)

        # Brightness
        brightness = float(np.mean(gray)) / 255.0
        if brightness < 0.05 or brightness > 0.97:
            brightness_score = 0.0
        elif brightness < self.min_brightness:
            brightness_score = (brightness - 0.05) / 0.05
        elif brightness > self.max_brightness:
            brightness_score = (0.97 - brightness) / 0.05
        else:
            brightness_score = 1.0

        # Face size
        h, w = face_crop.shape[:2]
        size_score = min(1.0, min(h, w) / (self.min_face_size * 2))

        return 0.4 * sharpness_score + 0.3 * brightness_score + 0.3 * size_score

    def breakdown(self, face_crop: np.ndarray) -> dict:
        """Return individual quality scores as a dict (for live UI feedback).

        Keys: overall, sharpness, brightness, brightness_raw, size, contrast
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, lap_var / 500.0)

        brightness_raw = float(np.mean(gray)) / 255.0
        if brightness_raw < 0.05 or brightness_raw > 0.97:
            brightness = 0.0
        elif brightness_raw < self.min_brightness:
            brightness = (brightness_raw - 0.05) / 0.05
        elif brightness_raw > self.max_brightness:
            brightness = (0.97 - brightness_raw) / 0.05
        else:
            brightness = 1.0

        h, w = face_crop.shape[:2]
        size = min(1.0, min(h, w) / (self.min_face_size * 2))

        contrast = min(1.0, float(gray.std()) / 64.0)

        overall = 0.4 * sharpness + 0.3 * brightness + 0.3 * size
        return {
            "overall": overall,
            "sharpness": sharpness,
            "brightness": brightness,
            "brightness_raw": brightness_raw,
            "size": size,
            "contrast": contrast,
        }

    def passes_all(self, face_crop: np.ndarray) -> bool:
        """Return True only if ALL quality gates pass."""
        return (
            self.is_large_enough(face_crop)
            and self.is_bright(face_crop)
            and self.is_sharp(face_crop)
        )

    def reject_reason(self, face_crop: np.ndarray) -> Optional[str]:
        """Return a human-readable reason for rejection, or None if the frame passes."""
        if not self.is_large_enough(face_crop):
            return "too_small"
        if not self.is_bright(face_crop):
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray)) / 255.0
            return "too_dark" if brightness < self.min_brightness else "overexposed"
        if not self.is_sharp(face_crop):
            return "blurry"
        return None
