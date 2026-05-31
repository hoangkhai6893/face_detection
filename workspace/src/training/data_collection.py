#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

import config
from core.encoder import rebuild_encodings, update_person_encodings
from core.frame_extractor import (
    ExtractedFrame,
    VideoFrameExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    person_name: str
    frames_processed: int
    frames_saved: int
    frames_rejected: int
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    avg_quality: float = 0.0


class DataCollectionSession:
    """Orchestrates a single data collection run for one person."""

    def __init__(
        self,
        person_name: str,
        extractor: VideoFrameExtractor,
        dataset_path: str = config.DATASET_PATH,
    ):
        self.person_name = person_name
        self.extractor = extractor
        self.person_folder = Path(dataset_path) / person_name
        self.person_folder.mkdir(parents=True, exist_ok=True)
        self._saved_frame_paths: List[str] = []  # track new files for incremental update

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_from_file(
        self,
        video_path: str,
        max_frames: int = 200,
    ) -> CollectionResult:
        """Process a video file and save high-quality face crops."""
        logger.info("Processing video: %s (max %d frames)", video_path, max_frames)
        start = time.time()

        qualities: List[float] = []
        reject_reasons: Dict[str, int] = {}
        frames_processed = 0
        frames_saved = 0

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Use tqdm over the generator for a progress bar
        with tqdm(
            total=min(max_frames, total_frames // self.extractor.frame_skip or max_frames),
            desc=f"  Collecting for {self.person_name}",
            unit="frame",
            ncols=80,
        ) as pbar:
            for ef in self.extractor.extract_from_file(video_path, max_frames):
                frames_saved += 1
                qualities.append(ef.quality_score)
                self._save_frame(ef)
                pbar.update(1)
                pbar.set_postfix(saved=frames_saved, q=f"{ef.quality_score:.2f}")

        # Estimate processed frames
        cap2 = cv2.VideoCapture(video_path)
        frames_processed = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()

        result = CollectionResult(
            person_name=self.person_name,
            frames_processed=frames_processed,
            frames_saved=frames_saved,
            frames_rejected=frames_processed - frames_saved,
            reject_reasons=reject_reasons,
            duration_seconds=time.time() - start,
            avg_quality=float(np.mean(qualities)) if qualities else 0.0,
        )
        if frames_saved > 0:
            self._rebuild_encodings()
        return result

    def run_from_camera(
        self,
        camera_id: int = 0,
        target_frames: int = 50,
        raw_video_dir: Optional[str] = None,
    ) -> CollectionResult:
        """Record a raw video from camera, then extract frames with retry logic.

        Flow:
          1. Record raw video (preview → SPACE to start → SPACE/Q/ESC to stop)
          2. Save .mp4 to *raw_video_dir* (default: <workspace>/data/)
          3. Extract frames via extract_with_retry (up to 3 passes with
             progressively relaxed thresholds until *target_frames* is reached)
          4. Rebuild encodings if any frames were saved
        """
        # Determine where to save raw recordings
        if raw_video_dir is None:
            workspace_root = Path(config.DATASET_PATH).parent
            raw_video_dir = str(workspace_root / "data")

        Path(raw_video_dir).mkdir(parents=True, exist_ok=True)

        timestamp_ms = int(time.time() * 1000)
        video_filename = f"{self.person_name}_{timestamp_ms}.mp4"
        output_path = str(Path(raw_video_dir) / video_filename)

        print(f"\n  Ghi video cho: {self.person_name}")
        print(f"  File se luu tai: {output_path}")
        print("  [SPACE = Bat dau ghi | SPACE / Q / ESC = Dung ghi]\n")

        # Phase 1: Record raw video from camera
        recorded_path = self.extractor.record_from_camera(camera_id, output_path)

        if recorded_path is None:
            logger.info("Camera recording cancelled — no frames collected")
            return CollectionResult(
                person_name=self.person_name,
                frames_processed=0,
                frames_saved=0,
                frames_rejected=0,
            )

        logger.info("Video saved: %s", recorded_path)
        print(f"\n  Video da luu: {recorded_path}")
        print(f"  Dang xu ly video (target: {target_frames} frames)...\n")

        # Phase 2: Extract frames with multi-pass retry
        start = time.time()
        qualities: List[float] = []
        frames_saved = 0

        cap = cv2.VideoCapture(recorded_path)
        frames_processed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        with tqdm(
            total=target_frames,
            desc=f"  Trich xuat frame",
            unit="frame",
            ncols=80,
        ) as pbar:
            for ef in self.extractor.extract_with_retry(recorded_path, target_frames):
                frames_saved += 1
                qualities.append(ef.quality_score)
                self._save_frame(ef)
                pbar.update(1)
                pbar.set_postfix(saved=frames_saved, q=f"{ef.quality_score:.2f}")

        result = CollectionResult(
            person_name=self.person_name,
            frames_processed=frames_processed,
            frames_saved=frames_saved,
            frames_rejected=max(0, frames_processed - frames_saved),
            duration_seconds=time.time() - start,
            avg_quality=float(np.mean(qualities)) if qualities else 0.0,
        )
        if frames_saved > 0:
            self._rebuild_encodings()
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_frame(self, ef: ExtractedFrame) -> str:
        timestamp_ms = int(time.time() * 1000)
        filename = f"{self.person_name}_{timestamp_ms}_q{ef.quality_score:.2f}.jpg"
        dest = self.person_folder / filename
        cv2.imwrite(str(dest), ef.face_crop)
        self._saved_frame_paths.append(str(dest))
        return str(dest)

    def _rebuild_encodings(self) -> None:
        encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        new_added, total = update_person_encodings(
            person_name=self.person_name,
            new_image_paths=self._saved_frame_paths,
            encodings_path=encodings_path,
            model_path=config.MODEL_PATH,
            logger=logger,
        )
        logger.info(
            "Incremental update done: +%d new encodings → %d total for '%s'",
            new_added, total, self.person_name,
        )
