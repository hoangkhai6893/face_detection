#!/usr/bin/env python3
"""
Frame Extractor — intelligent video frame selection for face training data.

Provides three components:
  FrameQualityChecker   — per-frame quality metrics (sharpness, brightness, size)
  FrameDiversityFilter  — SSIM-based deduplication across selected frames
  VideoFrameExtractor   — orchestrates extraction from video files or camera
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Set

import cv2
import face_recognition
import numpy as np
from skimage.metrics import structural_similarity as ssim

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.face_utils import extract_face_region

logger = logging.getLogger(__name__)

# How many video frames to skip between candidates (reduces processing load).
# At 30 fps, FRAME_SKIP=3 means one candidate every ~0.10 s.
FRAME_SKIP = 3


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFrame:
    """A face frame that passed all quality and diversity filters."""
    face_crop: np.ndarray     # BGR face crop (with padding)
    quality_score: float      # Combined quality score [0, 1]
    source_frame_num: int     # Frame index in the original video
    timestamp: float          # Seconds from the start of the video/session
    detection_conf: float     # YOLO confidence; 0.0 means HOG fallback was used


# ---------------------------------------------------------------------------
# Quality checker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class VideoFrameExtractor:
    """Extracts diverse, high-quality face frames from video files or camera.

    Detection pipeline per frame:
      1. YOLO detect → if no face: skip
      2. extract_face_region() (padding from config)
      3. FrameQualityChecker: size / brightness / sharpness gates
      4. FrameDiversityFilter: SSIM deduplication
      5. Yield ExtractedFrame
    """

    def __init__(
        self,
        yolo_model,
        quality_checker: Optional[FrameQualityChecker] = None,
        diversity_filter: Optional[FrameDiversityFilter] = None,
        frame_skip: int = FRAME_SKIP,
        detect_conf: float = config.FRAME_DETECT_CONF,
    ):
        self.yolo_model = yolo_model
        self.quality = quality_checker or FrameQualityChecker()
        self.diversity = diversity_filter or FrameDiversityFilter()
        self.frame_skip = frame_skip
        self.detect_conf = detect_conf

    # ------------------------------------------------------------------
    # Public interface — video file
    # ------------------------------------------------------------------

    def extract_from_file(
        self,
        video_path: str,
        max_frames: int = 200,
    ) -> Iterator[ExtractedFrame]:
        """Yield up to *max_frames* high-quality, diverse face frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video file: %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.diversity.reset()
        saved = 0
        frame_num = 0

        try:
            while saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                if frame_num % self.frame_skip != 0:
                    continue

                timestamp = frame_num / fps
                face_crop = self._detect_best_face(frame)
                if face_crop is None:
                    continue

                reason = self.quality.reject_reason(face_crop)
                if reason is not None:
                    continue

                if not self.diversity.is_diverse(face_crop, frame_num):
                    continue

                self.diversity.accept(face_crop, frame_num)
                saved += 1
                yield ExtractedFrame(
                    face_crop=face_crop,
                    quality_score=self.quality.score(face_crop),
                    source_frame_num=frame_num,
                    timestamp=timestamp,
                    detection_conf=0.0,
                )
        finally:
            cap.release()

    def extract_with_retry(
        self,
        video_path: str,
        target_frames: int = 50,
        max_frames: int = 200,
    ) -> Iterator[ExtractedFrame]:
        """Extract frames from a video, re-reading with relaxed thresholds if needed.

        Pass 1: default diversity thresholds (ssim=0.85, gap=15)
        Pass 2: relaxed (ssim=0.92, gap=5)  — skips already-saved frame numbers
        Pass 3: minimal (ssim=0.99, gap=1)  — skips already-saved frame numbers

        This allows the system to fill gaps when a short recording doesn't yield
        enough diverse frames at strict thresholds.
        """
        saved_frame_nums: Set[int] = set()
        saved = 0

        # Pass 1 — default thresholds
        logger.info("Extract pass 1 (default thresholds): target=%d", target_frames)
        for ef in self.extract_from_file(video_path, target_frames):
            saved_frame_nums.add(ef.source_frame_num)
            saved += 1
            yield ef

        if saved >= target_frames:
            return

        # Pass 2 — relaxed diversity
        remaining = target_frames - saved
        logger.info(
            "Extract pass 2 (relaxed thresholds): %d/%d, need %d more",
            saved, target_frames, remaining,
        )
        for ef in self._extract_internal(
            video_path, remaining, ssim_threshold=0.92, min_frame_gap=5,
            skip_frame_nums=saved_frame_nums,
        ):
            saved_frame_nums.add(ef.source_frame_num)
            saved += 1
            yield ef

        if saved >= target_frames:
            return

        # Pass 3 — minimal diversity filter
        remaining = target_frames - saved
        logger.info(
            "Extract pass 3 (minimal filter): %d/%d, need %d more",
            saved, target_frames, remaining,
        )
        for ef in self._extract_internal(
            video_path, remaining, ssim_threshold=0.99, min_frame_gap=1,
            skip_frame_nums=saved_frame_nums,
        ):
            yield ef

    # ------------------------------------------------------------------
    # Public interface — camera recording
    # ------------------------------------------------------------------

    def record_from_camera(
        self,
        camera_id: int = 0,
        output_path: str = "",
    ) -> Optional[str]:
        """Record raw video from a live camera and save it to disk.

        Two-phase flow:
          Phase 1 — PREVIEW: camera is open, user positions themselves.
                    SPACE = start recording, ESC/Q = cancel.
          Phase 2 — RECORDING: writes raw frames with cv2.VideoWriter.
                    SPACE / Q / ESC = stop recording.

        Returns the path to the saved video file, or None if cancelled.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Cannot open camera %d", camera_id)
            return None

        # Read actual camera capabilities — do NOT force resolution via cap.set()
        # because forcing unsupported resolutions causes black frames on many cameras.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        # Warm-up: read and discard frames so the sensor auto-exposes correctly.
        # 10 frames at 30fps ≈ 330ms — enough for most USB cameras to stabilise.
        for _ in range(10):
            cap.read()

        writer: Optional[cv2.VideoWriter] = None
        recording = False
        record_start: float = 0.0
        frame_count = 0

        # Plain ASCII window name — em-dash characters can confuse some Qt5 builds.
        win = "Face Collection Camera"

        # Create the window before the loop so Qt has time to initialise it.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, width, height)

        # Quality assessment state — updated every QUALITY_INTERVAL frames so
        # YOLO (~120 ms) does not block every single display refresh.
        quality_cache: Optional[dict] = None
        QUALITY_INTERVAL = 6   # assess face quality every N frames
        loop_counter = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                loop_counter += 1

                # ── Quality assessment (throttled) ─────────────────────────
                # Run on every Nth frame.  During recording we write the raw
                # frame BEFORE YOLO so no recorded frames are ever dropped.
                if loop_counter % QUALITY_INTERVAL == 0:
                    quality_cache = self._assess_frame_quality(frame)

                # Always work on a copy so the raw frame is unchanged for the writer.
                display = frame.copy()

                if not recording:
                    # ── Phase 1: PREVIEW ────────────────────────────────────
                    # waitKey(30) gives Qt5 enough time to paint the window
                    # when there is no GPU-accelerated HighGUI.
                    self._draw_preview_overlay(display)
                    self._draw_quality_feedback(display, quality_cache)
                    cv2.imshow(win, display)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        recording = True
                        record_start = time.time()
                        logger.info("Recording started -> %s", output_path)
                    elif key in (27, ord('q')):
                        logger.info("Camera collection cancelled")
                        return None
                else:
                    # ── Phase 2: RECORDING ───────────────────────────────────
                    # Write raw frame first — before any heavy processing —
                    # so the video file always gets the captured frame even
                    # when YOLO assessment blocks for ~120 ms on this iteration.
                    writer.write(frame)
                    frame_count += 1
                    elapsed = time.time() - record_start
                    self._draw_recording_raw_overlay(display, frame_count, elapsed)
                    self._draw_quality_feedback(display, quality_cache)
                    cv2.imshow(win, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord(' '), ord('q'), 27):
                        logger.info(
                            "Recording stopped: %d frames (%.1fs)", frame_count, elapsed
                        )
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        if frame_count == 0:
            logger.warning("No frames recorded — output file not created")
            return None

        return output_path

    # ------------------------------------------------------------------
    # Internal — frame processing
    # ------------------------------------------------------------------

    def _detect_best_face(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return the best face crop from YOLO (or HOG fallback), or None."""
        crop, _, _ = self._detect_face(frame)
        if crop is None:
            crop, _, _ = self._detect_face_hog(frame)
        return crop

    def _extract_internal(
        self,
        video_path: str,
        max_frames: int,
        ssim_threshold: float,
        min_frame_gap: int,
        skip_frame_nums: Set[int],
    ) -> Iterator[ExtractedFrame]:
        """Read *video_path* with a temporary diversity filter and custom thresholds.

        Frames whose source_frame_num is in *skip_frame_nums* are skipped so
        we don't save the same frame twice across retry passes.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        temp_diversity = FrameDiversityFilter(ssim_threshold, min_frame_gap)
        saved = 0
        frame_num = 0

        try:
            while saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                if frame_num % self.frame_skip != 0:
                    continue
                if frame_num in skip_frame_nums:
                    continue

                timestamp = frame_num / fps
                face_crop = self._detect_best_face(frame)
                if face_crop is None:
                    continue

                reason = self.quality.reject_reason(face_crop)
                if reason is not None:
                    continue

                if not temp_diversity.is_diverse(face_crop, frame_num):
                    continue

                temp_diversity.accept(face_crop, frame_num)
                saved += 1
                yield ExtractedFrame(
                    face_crop=face_crop,
                    quality_score=self.quality.score(face_crop),
                    source_frame_num=frame_num,
                    timestamp=timestamp,
                    detection_conf=0.0,
                )
        finally:
            cap.release()

    def _detect_face(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple], float]:
        """Run YOLO detection and return (crop, bbox, conf) or (None, None, 0.0)."""
        try:
            results = self.yolo_model(frame, verbose=False)
            best_conf = self.detect_conf
            best_crop = None
            best_box = None

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf <= best_conf:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    crop = extract_face_region(
                        frame, x1, y1, x2, y2, config.LEARNING_PADDING
                    )
                    if crop is not None:
                        best_conf = conf
                        best_crop = crop
                        best_box = (x1, y1, x2, y2)

            return best_crop, best_box, best_conf if best_crop is not None else 0.0
        except Exception as e:
            logger.debug("YOLO detection error: %s", e)
            return None, None, 0.0

    def _detect_face_hog(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple], float]:
        """Fallback: face_recognition HOG detector."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                return None, None, 0.0
            top, right, bottom, left = max(
                locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
            )
            x1, y1, x2, y2 = left, top, right, bottom
            crop = extract_face_region(
                frame, x1, y1, x2, y2, config.LEARNING_PADDING
            )
            return crop, (x1, y1, x2, y2), 0.0
        except Exception as e:
            logger.debug("HOG fallback error: %s", e)
            return None, None, 0.0

    # ------------------------------------------------------------------
    # Internal — quality assessment for live camera feedback
    # ------------------------------------------------------------------

    def _assess_frame_quality(self, frame: np.ndarray) -> Optional[dict]:
        """Detect the best face in *frame* and return a quality summary dict.

        Returns None if no face is detected.

        Dict keys: bbox, conf, overall, sharpness, brightness, brightness_raw,
                   size, contrast, guidance, guidance_color
        """
        face_crop, bbox, conf = self._detect_face(frame)
        if face_crop is None:
            face_crop, bbox, conf = self._detect_face_hog(frame)
        if face_crop is None:
            return None

        scores = self.quality.breakdown(face_crop)

        # Choose guidance text + BGR color based on weakest metric
        reason = self.quality.reject_reason(face_crop)
        if reason == "too_small":
            guidance = "Mat qua nho - den gan camera hon"
            g_color = (0, 100, 255)
        elif reason == "too_dark":
            guidance = "Qua toi - can them anh sang"
            g_color = (0, 100, 255)
        elif reason == "overexposed":
            guidance = "Qua sang - tranh nguon sang truc tiep"
            g_color = (0, 100, 255)
        elif reason == "blurry":
            guidance = "Bi mo - giu yen va nhin thang camera"
            g_color = (0, 100, 255)
        elif scores["overall"] >= 0.70:
            guidance = "Chat luong tot!  Giu nguyen tu the nay"
            g_color = (0, 220, 0)
        elif scores["overall"] >= 0.50:
            guidance = "Chap nhan duoc - co the dieu chinh them"
            g_color = (0, 200, 255)
        else:
            guidance = "Chat luong kem - dieu chinh anh sang & khoang cach"
            g_color = (0, 50, 220)

        return {
            "bbox": bbox,
            "conf": conf,
            "overall": scores["overall"],
            "sharpness": scores["sharpness"],
            "brightness": scores["brightness"],
            "brightness_raw": scores["brightness_raw"],
            "size": scores["size"],
            "contrast": scores["contrast"],
            "guidance": guidance,
            "guidance_color": g_color,
        }

    # ------------------------------------------------------------------
    # Internal — camera UI overlays
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_quality_feedback(frame: np.ndarray, quality: Optional[dict]) -> None:
        """Draw face bbox + bottom quality panel on the display frame (in-place).

        Layout (bottom 80 px, semi-transparent dark panel):
          ┌──────────────────────────────────────────────┐
          │ [████████░░] Chat luong tong: 72%            │
          │ Sac:0.85  Sang:0.70  Mat:0.65  Tuong:0.80   │
          │ Chat luong tot!  Giu nguyen tu the nay        │
          └──────────────────────────────────────────────┘
        """
        h, w = frame.shape[:2]
        PANEL_H = 80
        panel_top = h - PANEL_H

        # Semi-transparent dark panel at bottom
        frame[panel_top:] = (
            frame[panel_top:].astype(np.float32) * 0.28
        ).clip(0, 255).astype(np.uint8)

        if quality is None:
            # No face — centered notice
            msg = "Khong thay khuon mat - dung truoc camera"
            (tw, _th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, msg, ((w - tw) // 2, panel_top + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 220), 2)
            return

        # ── Face bounding box ─────────────────────────────────────────
        score = quality["overall"]
        if score >= 0.70:
            bbox_color = (0, 220, 0)       # green
        elif score >= 0.50:
            bbox_color = (0, 200, 255)     # yellow
        else:
            bbox_color = (0, 60, 220)      # red/orange

        x1, y1, x2, y2 = quality["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

        # Confidence label above bbox
        lbl = f"Q:{score:.2f}  Det:{quality['conf']:.2f}"
        lbl_y = max(y1 - 8, 18)
        cv2.putText(frame, lbl, (x1, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, bbox_color, 2)

        # ── Overall quality bar ───────────────────────────────────────
        bx, by = 10, panel_top + 8
        bw, bh = w - 20, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        filled = int(bw * min(score, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), bbox_color, -1)
        bar_label = f"Chat luong tong: {score:.0%}"
        cv2.putText(frame, bar_label, (bx + 5, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # ── Individual metric scores ──────────────────────────────────
        metrics_line = (
            f"Sac:{quality['sharpness']:.2f}   "
            f"Sang:{quality['brightness']:.2f}   "
            f"Mat:{quality['size']:.2f}   "
            f"Tuong:{quality['contrast']:.2f}"
        )
        cv2.putText(frame, metrics_line, (10, panel_top + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # ── Guidance text ─────────────────────────────────────────────
        cv2.putText(frame, quality["guidance"], (10, panel_top + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, quality["guidance_color"], 2)

    @staticmethod
    def _draw_preview_overlay(frame: np.ndarray) -> None:
        """Phase 1 overlay: top bar with preview instructions."""
        w = frame.shape[1]
        BAR_H = 50
        frame[:BAR_H] = (frame[:BAR_H].astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
        cv2.putText(
            frame, "PREVIEW  |  SPACE = Bat dau ghi   ESC/Q = Huy",
            (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2,
        )

    @staticmethod
    def _draw_recording_raw_overlay(
        frame: np.ndarray,
        frame_count: int,
        elapsed: float,
    ) -> None:
        """Phase 2 overlay: top bar with REC indicator and timer."""
        h, w = frame.shape[:2]
        BAR_H = 50
        frame[:BAR_H] = (frame[:BAR_H].astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)

        # Red ● REC dot
        cv2.circle(frame, (18, 25), 10, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (32, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

        # Duration | frame count
        mins, secs = divmod(int(elapsed), 60)
        status = f"{mins:02d}:{secs:02d}  |  {frame_count} frames  |  SPACE/Q/ESC = Dung"
        cv2.putText(frame, status, (90, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
