#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import config

_SETTINGS_FILE = Path(config.DATASET_PATH).parent / "extraction_settings.json"


@dataclass
class ExtractionSettings:
    """All tunable parameters for frame extraction strictness.

    Persisted to *_SETTINGS_FILE* so the user does not have to re-tune every
    session. Changing any value here affects the next call to _get_extractor().
    """
    min_laplacian: float = config.FRAME_MIN_LAPLACIAN
    min_face_px: int = config.FRAME_MIN_FACE_PX
    detect_conf: float = config.FRAME_DETECT_CONF
    ssim_threshold: float = 0.85
    min_frame_gap: int = 10
    frame_skip: int = 3

    def save(self, path: Path = _SETTINGS_FILE) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "min_laplacian": self.min_laplacian,
                "min_face_px": self.min_face_px,
                "detect_conf": self.detect_conf,
                "ssim_threshold": self.ssim_threshold,
                "min_frame_gap": self.min_frame_gap,
                "frame_skip": self.frame_skip,
            }, fh, indent=2)

    @classmethod
    def load(cls, path: Path = _SETTINGS_FILE) -> "ExtractionSettings":
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            defaults = cls()
            return cls(
                min_laplacian=float(data.get("min_laplacian", defaults.min_laplacian)),
                min_face_px=int(data.get("min_face_px", defaults.min_face_px)),
                detect_conf=float(data.get("detect_conf", defaults.detect_conf)),
                ssim_threshold=float(data.get("ssim_threshold", defaults.ssim_threshold)),
                min_frame_gap=int(data.get("min_frame_gap", defaults.min_frame_gap)),
                frame_skip=int(data.get("frame_skip", defaults.frame_skip)),
            )
        except Exception:
            return cls()


# Three built-in presets — "Lenient" through "Strict"
EXTRACTION_PRESETS: Dict[str, ExtractionSettings] = {
    "Thoai mai — bat nhieu frame, it reject nhat": ExtractionSettings(
        min_laplacian=5, min_face_px=50, detect_conf=0.08,
        ssim_threshold=0.93, min_frame_gap=5, frame_skip=2,
    ),
    "Binh thuong — khuyen nghi (mac dinh)": ExtractionSettings(),
    "Khat khe — chi lay frame chat luong cao": ExtractionSettings(
        min_laplacian=40, min_face_px=120, detect_conf=0.30,
        ssim_threshold=0.78, min_frame_gap=20, frame_skip=6,
    ),
}
