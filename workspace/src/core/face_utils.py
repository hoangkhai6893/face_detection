#!/usr/bin/env python3
"""
Shared face-region extraction utility.
Used by improved_hybrid_recognition.py, face_learning_mode.py, and optimize_encodings.py
to avoid duplicating the same bounding-box + padding logic in every module.
"""

from typing import Optional, Tuple
import numpy as np


def extract_face_region(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: int,
) -> Optional[np.ndarray]:
    """
    Crop a face region from *frame* with pixel *padding* on all sides.

    Coordinates are clamped to the frame boundary, so no out-of-bounds
    access can occur regardless of bounding-box placement.

    Returns the cropped NumPy array, or None if the resulting crop is empty.
    """
    height, width = frame.shape[:2]
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(width, x2 + padding)
    y2_pad = min(height, y2 + padding)
    region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    return region if region.size > 0 else None
