#!/usr/bin/env python3
"""
RecognitionStabilizer — denoising layer for face recognition events.

Raw recognition output from FaceRecognizer is noisy: the first cache-miss
for a face often returns "Unknown" even for family members (bad angle, motion
blur at entry). This module tracks each face across multiple cache-miss events
and only emits a StableEvent once the result is consistent enough.

Thresholds:
  - Known person: >= STABILIZER_MIN_KNOWN votes out of last STABILIZER_WINDOW
    (default 2/5 — fast ~1 s, good for turning on lights)
  - Unknown:      >= STABILIZER_MIN_UNKNOWN votes out of last STABILIZER_WINDOW
    (default 4/5 — conservative ~4 s, avoids false alarms)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config

# Regex to parse "Name (0.91)" format returned by recognize_face_in_region()
_KNOWN_RE = re.compile(r"^(.+?)\s+\((\d+\.\d+)\)$")

# Strings that indicate a non-recognition event (not truly "Unknown stranger")
_NON_EVENTS = {"No Face", "Error", "No Training Data"}


@dataclass
class StableEvent:
    """Emitted when a face's identity becomes stable (confirmed over N votes)."""
    person_name: str            # "Khai" or "Unknown"
    confidence: Optional[float] # avg confidence if known, None if Unknown
    bbox: tuple                 # (x1, y1, x2, y2)
    frame: np.ndarray           # clean frame at the moment of stabilisation
    is_known: bool


@dataclass
class _FaceTrack:
    """Internal state for one tracked face."""
    bbox: tuple
    history: deque = field(default_factory=deque)   # raw name strings
    stable_name: Optional[str] = None               # last confirmed stable result
    last_frame: int = 0


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) bounding boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _parse_name(raw: str) -> tuple[str, Optional[float]]:
    """
    Parse recognize_face_in_region() output.
    "Khai (0.91)" → ("Khai", 0.91)
    "Unknown"     → ("Unknown", None)
    """
    m = _KNOWN_RE.match(raw)
    if m:
        return m.group(1), float(m.group(2))
    return raw, None


class RecognitionStabilizer:
    """
    Smooths noisy recognition results by requiring consistent identification
    across multiple cache-miss events before emitting a StableEvent.

    One instance per FaceRecognizer session. Call update() on every cache-miss.
    """

    def __init__(
        self,
        window_size: int = config.STABILIZER_WINDOW,
        min_count_known: int = config.STABILIZER_MIN_KNOWN,
        min_count_unknown: int = config.STABILIZER_MIN_UNKNOWN,
        iou_threshold: float = 0.4,
        track_timeout_frames: int = config.STABILIZER_TRACK_TIMEOUT,
    ) -> None:
        self.window_size = window_size
        self.min_count_known = min_count_known
        self.min_count_unknown = min_count_unknown
        self.iou_threshold = iou_threshold
        self.track_timeout = track_timeout_frames
        self._tracks: List[_FaceTrack] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_name: str,
        bbox: tuple,
        frame_num: int,
        frame: np.ndarray,
    ) -> Optional[StableEvent]:
        """
        Record a new cache-miss recognition for the face at *bbox*.

        Returns a StableEvent if the face's identity just became stable
        (and differs from its previous stable state). Returns None otherwise.

        raw_name: string from recognize_face_in_region(), e.g. "Khai (0.91)"
        bbox: (x1, y1, x2, y2) in original frame coordinates
        frame_num: self._frame_count from FaceRecognizer
        frame: BGR frame BEFORE draw_results() is called (already .copy()-ed by caller)
        """
        # Ignore transient errors — not security-relevant
        if raw_name in _NON_EVENTS:
            return None

        self._prune_stale(frame_num)

        track = self._find_or_create(bbox, frame_num)
        track.bbox = bbox
        track.last_frame = frame_num
        track.history.append(raw_name)

        return self._check_stable(track, frame)

    def reset(self) -> None:
        """Clear all tracked faces (e.g. between sessions)."""
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_stale(self, frame_num: int) -> None:
        self._tracks = [
            t for t in self._tracks
            if frame_num - t.last_frame <= self.track_timeout
        ]

    def _find_or_create(self, bbox: tuple, frame_num: int) -> _FaceTrack:
        for t in self._tracks:
            if _iou(t.bbox, bbox) > self.iou_threshold:
                return t
        track = _FaceTrack(
            bbox=bbox,
            history=deque(maxlen=self.window_size),
            last_frame=frame_num,
        )
        self._tracks.append(track)
        return track

    def _check_stable(self, track: _FaceTrack, frame: np.ndarray) -> Optional[StableEvent]:
        history = list(track.history)
        n = len(history)

        # Count Unknown votes
        unknown_count = sum(1 for h in history if h == "Unknown")
        known_names = [h for h in history if h not in ("Unknown",)]

        # Check Unknown stability (strict threshold)
        if unknown_count >= self.min_count_unknown:
            if track.stable_name != "Unknown":
                track.stable_name = "Unknown"
                return StableEvent(
                    person_name="Unknown",
                    confidence=None,
                    bbox=track.bbox,
                    frame=frame,
                    is_known=False,
                )
            return None

        # Count votes per known person
        if not known_names:
            return None

        name_votes: dict[str, list[float]] = {}
        for raw in known_names:
            parsed_name, conf = _parse_name(raw)
            if parsed_name not in name_votes:
                name_votes[parsed_name] = []
            if conf is not None:
                name_votes[parsed_name].append(conf)

        # Pick the person with the most votes (descending); ties keep current stable name.
        # This prevents flipping identity on a tied window (e.g. 2 Khai + 2 Lan → no change).
        candidates = sorted(name_votes.items(), key=lambda kv: len(kv[1]), reverse=True)
        for person, confs in candidates:
            if len(confs) >= self.min_count_known:
                if track.stable_name != person:
                    track.stable_name = person
                    avg_conf = float(np.mean(confs)) if confs else None
                    return StableEvent(
                        person_name=person,
                        confidence=avg_conf,
                        bbox=track.bbox,
                        frame=frame,
                        is_known=True,
                    )
                # Top candidate is already the stable person → no change
                return None

        return None
