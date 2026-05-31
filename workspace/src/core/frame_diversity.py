from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class FrameDiversityFilter:
    """Prevents near-duplicate frames from being saved.

    Two frames are considered duplicates when:
      - Their SSIM (on the face crop, resized to a common size) exceeds
        *ssim_threshold*, AND
      - The gap between their source frame numbers is smaller than
        *min_frame_gap*.
    """

    _COMPARE_SIZE = (64, 64)   # Resize crops to this before SSIM

    def __init__(self, ssim_threshold: float = 0.85, min_frame_gap: int = 10):
        self.ssim_threshold = ssim_threshold
        self.min_frame_gap = min_frame_gap
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_num: int = -9999

    def reset(self) -> None:
        """Reset state — call between independent collection sessions."""
        self._last_frame = None
        self._last_frame_num = -9999

    def is_diverse(self, face_crop: np.ndarray, frame_num: int) -> bool:
        """Return True if this frame is sufficiently different from the last accepted one."""
        if self._last_frame is None:
            return True   # First frame is always accepted

        # Minimum temporal gap check (fast, no image ops)
        if frame_num - self._last_frame_num < self.min_frame_gap:
            return False

        # SSIM check
        a = cv2.resize(cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY), self._COMPARE_SIZE)
        b = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), self._COMPARE_SIZE)
        similarity = float(ssim(a, b, data_range=255))
        return similarity < self.ssim_threshold

    def accept(self, face_crop: np.ndarray, frame_num: int) -> None:
        """Record this frame as the last accepted one."""
        self._last_frame = face_crop.copy()
        self._last_frame_num = frame_num
