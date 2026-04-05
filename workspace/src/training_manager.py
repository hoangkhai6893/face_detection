
#!/usr/bin/env python3

"""
Training Manager — unified data collection & person management tool.

Features:
  - Collect face data from a video file or live camera
  - Automatic frame selection (quality + diversity filters)
  - Interactive CLI: create / update / delete / reset persons
  - Optional augmentation and encoding optimization after collection

Usage:
    python training_manager.py
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import cv2
import numpy as np
import questionary
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Resolve imports regardless of working directory
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

import config
from core.encoder import rebuild_encodings
from core.frame_extractor import (
    ExtractedFrame,
    FrameDiversityFilter,
    FrameQualityChecker,
    VideoFrameExtractor,
)

_SETTINGS_FILE = Path(config.DATASET_PATH).parent / "extraction_settings.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonInfo:
    name: str
    image_count: int
    augmented_count: int
    folder_path: Path
    created_date: str


@dataclass
class CollectionResult:
    person_name: str
    frames_processed: int
    frames_saved: int
    frames_rejected: int
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    avg_quality: float = 0.0


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
_EXTRACTION_PRESETS: Dict[str, ExtractionSettings] = {
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


def _float_validator(lo: float, hi: float):
    """Return a questionary validator that checks float in [lo, hi]."""
    def _v(v: str) -> bool | str:
        try:
            fv = float(v)
        except ValueError:
            return "Nhap so thuc hop le"
        if lo <= fv <= hi:
            return True
        return f"Nhap so tu {lo} den {hi}"
    return _v


def _int_validator(lo: int, hi: int):
    """Return a questionary validator that checks int in [lo, hi]."""
    def _v(v: str) -> bool | str:
        try:
            iv = int(v)
        except ValueError:
            return "Nhap so nguyen hop le"
        if lo <= iv <= hi:
            return True
        return f"Nhap so nguyen tu {lo} den {hi}"
    return _v


# ---------------------------------------------------------------------------
# Person Manager — CRUD
# ---------------------------------------------------------------------------

_INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]|^\.|^\.\.')
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class PersonManager:
    """Create, list, delete, and reset person folders in the dataset."""

    def __init__(self, dataset_path: str = config.DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_persons(self) -> List[PersonInfo]:
        persons = []
        for entry in sorted(self.dataset_path.iterdir()):
            if not entry.is_dir():
                continue
            images = [
                f for f in entry.iterdir()
                if f.suffix.lower() in _IMAGE_EXTS
            ]
            aug = [f for f in images if "_aug_" in f.name]
            orig = [f for f in images if "_aug_" not in f.name]
            created = time.strftime(
                "%Y-%m-%d", time.localtime(entry.stat().st_ctime)
            )
            persons.append(PersonInfo(
                name=entry.name,
                image_count=len(orig),
                augmented_count=len(aug),
                folder_path=entry,
                created_date=created,
            ))
        return persons

    def get_image_count(self, name: str) -> int:
        folder = self.dataset_path / name
        if not folder.is_dir():
            return 0
        return sum(
            1 for f in folder.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and "_aug_" not in f.name
        )

    def get_augmented_count(self, name: str) -> int:
        folder = self.dataset_path / name
        if not folder.is_dir():
            return 0
        return sum(
            1 for f in folder.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and "_aug_" in f.name
        )

    def exists(self, name: str) -> bool:
        return (self.dataset_path / name).is_dir()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_person(self, name: str) -> Path:
        """Create a new person folder. Raises ValueError for invalid names."""
        if _INVALID_NAME_RE.search(name) or not name.strip():
            raise ValueError(f"Invalid person name: {name!r}")
        path = self.dataset_path / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Created person folder: %s", path)
        return path

    def delete_person(self, name: str, rebuild: bool = True) -> bool:
        """Delete a person folder and optionally rebuild encodings."""
        path = self.dataset_path / name
        if not path.is_dir():
            logger.warning("Person not found: %s", name)
            return False
        shutil.rmtree(path)
        logger.info("Deleted person: %s", name)
        if rebuild:
            self._rebuild()
        return True

    def reset_person(self, name: str, rebuild: bool = True) -> int:
        """Delete all images for a person (keep folder). Returns image count deleted."""
        path = self.dataset_path / name
        if not path.is_dir():
            logger.warning("Person not found: %s", name)
            return 0
        images = [
            f for f in path.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        ]
        for f in images:
            f.unlink()
        logger.info("Reset person %s: deleted %d images", name, len(images))
        if rebuild:
            self._rebuild()
        return len(images)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        logger.info("Rebuilding encodings → %s", encodings_path)
        total_enc, total_persons = rebuild_encodings(
            dataset_path=str(self.dataset_path),
            model_path=config.MODEL_PATH,
            encodings_path=encodings_path,
            logger=logger,
        )
        logger.info(
            "Encodings rebuilt — %d encodings for %d persons",
            total_enc, total_persons,
        )


# ---------------------------------------------------------------------------
# Data Collection Session
# ---------------------------------------------------------------------------

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
        return str(dest)

    def _rebuild_encodings(self) -> None:
        encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        logger.info("Rebuilding encodings → %s", encodings_path)
        total_enc, total_persons = rebuild_encodings(
            dataset_path=str(self.person_folder.parent),
            model_path=config.MODEL_PATH,
            encodings_path=encodings_path,
            logger=logger,
        )
        logger.info(
            "Encodings rebuilt — %d encodings for %d persons",
            total_enc, total_persons,
        )


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
_CREATE_NEW = "[ + Tạo người mới ]"


class InteractiveCLI:
    """Terminal UI using questionary for person management and data collection."""

    def __init__(self):
        self.person_mgr = PersonManager()
        self._yolo: Optional[YOLO] = None   # Lazy-loaded
        self.settings: ExtractionSettings = ExtractionSettings.load()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\n╔══════════════════════════════════════╗")
        print("║   TRAINING DATA MANAGER               ║")
        print("╚══════════════════════════════════════╝\n")

        while True:
            model_name = Path(config.MODEL_PATH).name
            choice = questionary.select(
                f"Chọn hành động:  [model: {model_name}]",
                choices=[
                    "1. Thu thập dữ liệu mới",
                    "2. Quản lý Persons (xem / xóa / reset)",
                    "3. Augment dataset",
                    "4. Rebuild encodings  (dùng model hiện tại)",
                    "5. Optimize encodings  (cluster + giảm số lượng)",
                    "6. Cài đặt thu thập frame",
                    "7. Hướng dẫn",
                    "8. Thoát",
                ],
                use_shortcuts=False,
            ).ask()

            if choice is None or choice.startswith("8"):
                print("Tạm biệt!")
                break
            elif choice.startswith("1"):
                self._collect_data_flow()
            elif choice.startswith("2"):
                self._manage_persons_flow()
            elif choice.startswith("3"):
                self._augment_flow()
            elif choice.startswith("4"):
                self._rebuild_flow()
            elif choice.startswith("5"):
                self._optimize_flow()
            elif choice.startswith("6"):
                self._settings_flow()
            elif choice.startswith("7"):
                self._help_flow()

    # ------------------------------------------------------------------
    # Data collection flow
    # ------------------------------------------------------------------

    def _collect_data_flow(self) -> None:
        print("\n--- Thu thập dữ liệu ---")

        # 1. Select or create person
        person_name = self._select_or_create_person()
        if person_name is None:
            return

        # 2. Select source
        source = questionary.select(
            "Nguồn dữ liệu:",
            choices=["Camera (real-time)", "Video file"],
        ).ask()
        if source is None:
            return

        # 3. Run collection
        extractor = self._get_extractor()
        session = DataCollectionSession(person_name, extractor)

        if source.startswith("Camera"):
            camera_id = self._select_camera()
            if camera_id is None:
                return
            target = questionary.text(
                "Số frame muốn thu thập:",
                default="50",
                validate=lambda v: v.isdigit() and int(v) > 0 or "Nhập số nguyên dương",
            ).ask()
            if target is None:
                return
            print(
                "\n  Luu y: He thong se ghi video truoc, sau do xu ly de trich xuat frame."
            )
            result = session.run_from_camera(camera_id, int(target))
        else:
            video_path = self._select_video_file()
            if video_path is None:
                return
            max_frames_str = questionary.text(
                "Số frame tối đa muốn lưu:",
                default="200",
                validate=lambda v: v.isdigit() and int(v) > 0 or "Nhập số nguyên dương",
            ).ask()
            if max_frames_str is None:
                return
            result = session.run_from_file(video_path, int(max_frames_str))

        # 4. Show summary
        self._show_collection_summary(result)

        # 5. Offer augment / optimize
        if result.frames_saved > 0:
            if questionary.confirm("Augment dataset ngay?", default=False).ask():
                self._augment_flow(person_name)
            if questionary.confirm("Optimize encodings ngay?", default=False).ask():
                self._optimize_flow()

    # ------------------------------------------------------------------
    # Person management flow
    # ------------------------------------------------------------------

    def _manage_persons_flow(self) -> None:
        print("\n--- Quản lý Persons ---")
        persons = self.person_mgr.list_persons()

        if not persons:
            print("Chưa có person nào trong dataset.")
            return

        # Print table
        print(f"\n{'Tên':<20} {'Ảnh gốc':>8} {'Augmented':>10} {'Ngày tạo':>12}")
        print("-" * 55)
        for p in persons:
            print(f"{p.name:<20} {p.image_count:>8} {p.augmented_count:>10} {p.created_date:>12}")
        print()

        person_names = [p.name for p in persons]
        selected = questionary.select(
            "Chọn person để thao tác (ESC để quay lại):",
            choices=person_names + ["[ Quay lại ]"],
        ).ask()

        if selected is None or selected == "[ Quay lại ]":
            return

        action = questionary.select(
            f"Thao tác với '{selected}':",
            choices=[
                "Reset (xóa ảnh, giữ folder)",
                "Xóa hoàn toàn (xóa cả folder)",
                "Quay lại",
            ],
        ).ask()

        if action is None or action == "Quay lại":
            return

        if action.startswith("Reset"):
            confirmed = questionary.confirm(
                f"Xóa toàn bộ ảnh của '{selected}'? (không thể hoàn tác)",
                default=False,
            ).ask()
            if confirmed:
                count = self.person_mgr.reset_person(selected)
                print(f"✓ Đã xóa {count} ảnh của '{selected}'.")

        elif action.startswith("Xóa hoàn toàn"):
            confirmed = questionary.confirm(
                f"Xóa HOÀN TOÀN '{selected}' khỏi dataset? (không thể hoàn tác)",
                default=False,
            ).ask()
            if confirmed:
                self.person_mgr.delete_person(selected)
                print(f"✓ Đã xóa '{selected}'.")

    # ------------------------------------------------------------------
    # Augment flow
    # ------------------------------------------------------------------

    def _rebuild_flow(self) -> None:
        """Rebuild toàn bộ face encodings từ dataset dùng model hiện tại."""
        print(f"\n--- Rebuild Encodings ---")
        print(f"  Model   : {config.MODEL_PATH}")
        print(f"  Dataset : {config.DATASET_PATH}")
        encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
        print(f"  Output  : {encodings_path}")

        persons = self.person_mgr.list_persons()
        if not persons:
            print("  Không có person nào trong dataset.")
            return

        print(f"\n  Persons: {', '.join(p.name for p in persons)}")
        total_images = sum(p.image_count + p.augmented_count for p in persons)
        print(f"  Tổng ảnh (gốc + augmented): {total_images:,}\n")

        confirmed = questionary.confirm(
            f"Rebuild encodings cho {len(persons)} persons ({total_images:,} ảnh)?",
            default=True,
        ).ask()
        if not confirmed:
            return

        import time
        start = time.time()
        print()
        total_enc, total_persons = rebuild_encodings(
            dataset_path=str(self.person_mgr.dataset_path),
            model_path=config.MODEL_PATH,
            encodings_path=encodings_path,
            logger=logger,
        )
        elapsed = time.time() - start
        print(f"\n  ✓ Xong — {total_enc} encodings | {total_persons} persons | {elapsed:.0f}s")

    def _augment_flow(self, person_name: Optional[str] = None) -> None:
        print("\n--- Augment Dataset ---")
        try:
            from augment_dataset import augment_person, AUGMENTATIONS
        except ImportError:
            logger.error("augment_dataset.py không tìm thấy trong sys.path")
            return

        if person_name:
            persons = [person_name]
        else:
            all_persons = self.person_mgr.list_persons()
            if not all_persons:
                print("Không có person nào.")
                return
            choices = [p.name for p in all_persons] + ["Tất cả"]
            sel = questionary.select("Augment cho ai?", choices=choices).ask()
            if sel is None:
                return
            persons = [p.name for p in all_persons] if sel == "Tất cả" else [sel]

        total_gen = 0
        total_skip = 0
        for pname in persons:
            gen, skip = augment_person(
                person_name=pname,
                dataset_path=config.DATASET_PATH,
                show_preview=False,
                logger=logger,
            )
            total_gen += gen
            total_skip += skip

        print(f"\n✓ Augmentation xong — generated: {total_gen} | skipped: {total_skip}")

    # ------------------------------------------------------------------
    # Optimize flow
    # ------------------------------------------------------------------

    def _optimize_flow(self) -> None:
        print("\n--- Optimize Encodings ---")
        try:
            from optimize_encodings import FaceEncodingOptimizer
        except ImportError:
            logger.error("optimize_encodings.py không tìm thấy")
            return

        optimizer = FaceEncodingOptimizer(config.DATASET_PATH, config.MODEL_PATH)
        success = optimizer.optimize_all_encodings()
        if success:
            optimizer.benchmark_performance()

    # ------------------------------------------------------------------
    # Helper UI methods
    # ------------------------------------------------------------------

    def _select_or_create_person(self) -> Optional[str]:
        persons = self.person_mgr.list_persons()
        choices = [f"{p.name}  ({p.image_count} ảnh)" for p in persons] + [_CREATE_NEW]

        selected = questionary.select("Chọn person:", choices=choices).ask()
        if selected is None:
            return None

        if selected == _CREATE_NEW:
            existing = {p.name for p in persons}
            name = questionary.text(
                "Nhập tên người mới:",
                validate=lambda v: (
                    "Tên không được để trống" if not v.strip()
                    else "Tên đã tồn tại" if v.strip() in existing
                    else "Tên không hợp lệ (chứa ký tự đặc biệt)" if _INVALID_NAME_RE.search(v)
                    else True
                ),
            ).ask()
            if name is None:
                return None
            name = name.strip()
            self.person_mgr.create_person(name)
            return name

        # Extract name from display string "Name  (N ảnh)"
        return selected.split("  ")[0]

    def _select_camera(self) -> Optional[int]:
        # Probe up to 3 camera indices
        available = []
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(f"Camera {idx}")
                cap.release()

        if not available:
            print("Không tìm thấy camera nào.")
            return None
        if len(available) == 1:
            return 0

        sel = questionary.select("Chọn camera:", choices=available).ask()
        if sel is None:
            return None
        return int(sel.split()[-1])

    def _select_video_file(self) -> Optional[str]:
        path = questionary.path(
            "Đường dẫn đến video file:",
            validate=lambda v: (
                "File không tồn tại" if not os.path.isfile(v)
                else "Định dạng không hỗ trợ (dùng mp4/avi/mov/mkv)" if Path(v).suffix.lower() not in _VIDEO_EXTS
                else True
            ),
        ).ask()
        return path

    def _get_extractor(self) -> VideoFrameExtractor:
        if self._yolo is None:
            logger.info("Loading YOLO model: %s", config.MODEL_PATH)
            self._yolo = YOLO(config.MODEL_PATH)
        s = self.settings
        return VideoFrameExtractor(
            yolo_model=self._yolo,
            quality_checker=FrameQualityChecker(
                min_laplacian=s.min_laplacian,
                min_face_size=s.min_face_px,
            ),
            diversity_filter=FrameDiversityFilter(
                ssim_threshold=s.ssim_threshold,
                min_frame_gap=s.min_frame_gap,
            ),
            frame_skip=s.frame_skip,
            detect_conf=s.detect_conf,
        )

    # ------------------------------------------------------------------
    # Settings flow
    # ------------------------------------------------------------------

    def _settings_flow(self) -> None:
        """Interactive menu for viewing and adjusting extraction parameters."""
        print("\n--- Cai dat Thu Thap Frame ---")
        while True:
            s = self.settings
            print(f"\n  Cai dat hien tai:")
            print(f"  {'Thong so':<35} {'Gia tri':>8}   Mo ta")
            print(f"  {'-'*65}")
            print(f"  {'min_laplacian (do sac net)':<35} {s.min_laplacian:>8}   "
                  f"[0-200] thap=de pass, cao=chi frame net")
            print(f"  {'min_face_px (kich thuoc mat toi thieu)':<35} {s.min_face_px:>8}px  "
                  f"[20-300] thap=bat mat xa/nghieng")
            print(f"  {'detect_conf (YOLO confidence)':<35} {s.detect_conf:>8.2f}  "
                  f"[0.05-0.9] thap=bat nhieu goc do hon")
            print(f"  {'ssim_threshold (tranh trung lap)':<35} {s.ssim_threshold:>8.2f}  "
                  f"[0.5-0.99] cao=chap nhan frame giong hon")
            print(f"  {'min_frame_gap (khoang cach frame)':<35} {s.min_frame_gap:>8}   "
                  f"[1-100] khoang cach toi thieu giua 2 lan luu")
            print(f"  {'frame_skip (bo qua moi N frame)':<35} {s.frame_skip:>8}   "
                  f"[1-30] thap=quet ky hon, cham hon")
            print()

            action = questionary.select(
                "Thao tac:",
                choices=[
                    "Ap dung preset",
                    "Tinh chinh thu cong",
                    "Dat lai ve mac dinh",
                    "Luu va quay lai",
                ],
            ).ask()

            if action is None or action == "Luu va quay lai":
                self.settings.save()
                print(f"  Cai dat da luu vao: {_SETTINGS_FILE}")
                break
            elif action == "Ap dung preset":
                preset_name = questionary.select(
                    "Chon preset:",
                    choices=list(_EXTRACTION_PRESETS.keys()),
                ).ask()
                if preset_name:
                    self.settings = _EXTRACTION_PRESETS[preset_name]
                    print(f"  Da ap dung: {preset_name}")
            elif action == "Dat lai ve mac dinh":
                self.settings = ExtractionSettings()
                print("  Da dat lai ve mac dinh.")
            elif action == "Tinh chinh thu cong":
                self._manual_tune_settings()

    def _manual_tune_settings(self) -> None:
        """Let the user edit individual extraction parameters one by one."""
        s = self.settings
        while True:
            param = questionary.select(
                "Chon thong so can dieu chinh:",
                choices=[
                    f"min_laplacian  (do sac net)          [{s.min_laplacian}]",
                    f"min_face_px    (kich thuoc mat toi thieu px) [{s.min_face_px}]",
                    f"detect_conf    (YOLO confidence)     [{s.detect_conf}]",
                    f"ssim_threshold (nguong trung lap)    [{s.ssim_threshold}]",
                    f"min_frame_gap  (khoang cach frame)   [{s.min_frame_gap}]",
                    f"frame_skip     (bo qua moi N frame)  [{s.frame_skip}]",
                    "Xong",
                ],
            ).ask()

            if param is None or param.startswith("Xong"):
                break
            elif param.startswith("min_laplacian"):
                val = questionary.text(
                    "min_laplacian [0-200]  (video nen mp4: 5-20 | camera that: 50-150):",
                    default=str(s.min_laplacian),
                    validate=_float_validator(0, 200),
                ).ask()
                if val is not None:
                    s.min_laplacian = float(val)
            elif param.startswith("min_face_px"):
                val = questionary.text(
                    "min_face_px [20-300]  (20=cho phep mat rat nho | 100=chi mat lon):",
                    default=str(s.min_face_px),
                    validate=_int_validator(20, 300),
                ).ask()
                if val is not None:
                    s.min_face_px = int(val)
            elif param.startswith("detect_conf"):
                val = questionary.text(
                    "detect_conf [0.05-0.90]  (0.05=bat ca mat nghieng | 0.50=chi mat ro):",
                    default=str(s.detect_conf),
                    validate=_float_validator(0.05, 0.90),
                ).ask()
                if val is not None:
                    s.detect_conf = float(val)
            elif param.startswith("ssim_threshold"):
                val = questionary.text(
                    "ssim_threshold [0.50-0.99]  (0.99=chi bo qua frame giong het | 0.70=loc kenh hon):",
                    default=str(s.ssim_threshold),
                    validate=_float_validator(0.50, 0.99),
                ).ask()
                if val is not None:
                    s.ssim_threshold = float(val)
            elif param.startswith("min_frame_gap"):
                val = questionary.text(
                    "min_frame_gap [1-100]  (1=luu lien tiep | 15=cach 15 frame moi luu):",
                    default=str(s.min_frame_gap),
                    validate=_int_validator(1, 100),
                ).ask()
                if val is not None:
                    s.min_frame_gap = int(val)
            elif param.startswith("frame_skip"):
                val = questionary.text(
                    "frame_skip [1-30]  (1=quet moi frame | 5=bo qua 4, quet 1):",
                    default=str(s.frame_skip),
                    validate=_int_validator(1, 30),
                ).ask()
                if val is not None:
                    s.frame_skip = int(val)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _help_flow(self) -> None:
        """Interactive help system."""
        _HELP_TOPICS = {
            "Tổng quan — Quy trình đầy đủ": _help_overview,
            "1. Thu thập dữ liệu (camera / video)": _help_collect,
            "2. Augment dataset": _help_augment,
            "3. Rebuild encodings": _help_rebuild,
            "4. Optimize encodings": _help_optimize,
            "5. Cài đặt thu thập frame (tham số)": _help_settings,
            "← Quay lại menu chính": None,
        }
        while True:
            print()
            topic = questionary.select(
                "Hướng dẫn — chọn chủ đề:",
                choices=list(_HELP_TOPICS.keys()),
                use_shortcuts=False,
            ).ask()
            if topic is None or _HELP_TOPICS[topic] is None:
                break
            _HELP_TOPICS[topic]()
            input("\n  [Enter để tiếp tục...]")

    @staticmethod
    def _show_collection_summary(result: CollectionResult) -> None:
        print(f"\n{'─' * 45}")
        print(f"  Kết quả thu thập: {result.person_name}")
        print(f"{'─' * 45}")
        print(f"  Frames đã lưu:   {result.frames_saved}")
        print(f"  Frames bị bỏ:    {result.frames_rejected}")
        print(f"  Chất lượng tb:   {result.avg_quality:.2f}")
        print(f"  Thời gian:       {result.duration_seconds:.1f}s")
        if result.reject_reasons:
            print("  Lý do bị bỏ:")
            for reason, count in result.reject_reasons.items():
                print(f"    {reason}: {count}")
        print(f"{'─' * 45}\n")


# ---------------------------------------------------------------------------
# Help text functions (called by _help_flow)
# ---------------------------------------------------------------------------

def _help_overview() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║              QUY TRÌNH ĐẦY ĐỦ — NGƯỜI MỚI BẮT ĐẦU          ║
╚══════════════════════════════════════════════════════════════╝

Bước 1 — Thu thập ảnh khuôn mặt
  Menu → "1. Thu thập dữ liệu mới"
  ├── Chọn / tạo person (ví dụ: "Khai")
  ├── Chọn nguồn: Camera hoặc Video file
  └── Hệ thống tự động chọn frame đa dạng, chất lượng

Bước 2 — (Tùy chọn) Augment để tăng dữ liệu
  Menu → "3. Augment dataset"
  Tạo thêm 8 biến thể cho mỗi ảnh gốc:
  mờ nhẹ, mờ mạnh, tối nhẹ, tối nặng, nhiễu, v.v.
  → Dùng khi có ÍT ảnh gốc (< 20 ảnh/người)

Bước 3 — Rebuild encodings (huấn luyện lại)
  Menu → "4. Rebuild encodings"
  Đọc toàn bộ ảnh trong dataset → trích xuất vector khuôn mặt
  → Bắt buộc sau khi thêm ảnh mới / đổi model YOLO

Bước 4 — (Tùy chọn) Optimize để giảm số lượng
  Menu → "5. Optimize encodings"
  Gộp các encoding giống nhau → giữ tối đa 30/người
  → Tăng tốc nhận diện khi có nhiều ảnh

Bước 5 — Chạy nhận diện
  python src/recognizer.py

Lưu ý về model YOLO (config.py → MODEL_PATH):
  yolov11n  → nhanh nhất (17 FPS), phát hiện tốt
  yolov12s  → cân bằng (7 FPS), chính xác hơn
  yolov12l  → chậm (2 FPS), dùng khi CPU mạnh / GPU
""")


def _help_collect() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║              THU THẬP DỮ LIỆU — CAMERA / VIDEO              ║
╚══════════════════════════════════════════════════════════════╝

Hai chế độ:

  [Camera]
  • Mở webcam, thu thập real-time
  • Hiển thị live preview: bbox, chất lượng, số frame đã lưu
  • Nhấn Q để dừng sớm, ESC để hủy
  • Khuyến nghị: target 30-50 frame/người

  [Video file]
  • Nhập đường dẫn file .mp4 / .avi / .mov / .mkv
  • Hệ thống quét video, tự chọn frame đa dạng
  • Không cần ngồi chờ — xử lý tự động

Tiêu chí frame được lưu (tất cả phải đạt):
  ✓ Phát hiện được khuôn mặt (YOLO confidence ≥ detect_conf)
  ✓ Khuôn mặt đủ lớn (min_face_px)
  ✓ Ảnh đủ nét (Laplacian variance ≥ min_laplacian)
  ✓ Độ sáng hợp lệ (10% - 92% — không quá tối / quá sáng)
  ✓ Đủ khác biệt với frame trước (SSIM < ssim_threshold)
  ✓ Cách frame trước đủ xa (min_frame_gap)

Lý do frame bị bỏ:
  no_face      — YOLO không tìm thấy khuôn mặt
  too_small    — khuôn mặt quá nhỏ (đứng xa camera)
  blurry       — ảnh bị mờ / chuyển động nhanh
  too_dark/bright — ánh sáng không phù hợp
  duplicate    — frame quá giống frame vừa lưu

Gợi ý quay video tốt:
  • Ánh sáng đủ, không ngược sáng
  • Quay nhiều góc: thẳng, nghiêng trái/phải, cúi/ngửa nhẹ
  • Di chuyển chậm, tránh chuyển động nhanh
  • Khoảng cách 50-200cm tới camera
""")


def _help_augment() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    AUGMENT DATASET                           ║
╚══════════════════════════════════════════════════════════════╝

Augment tạo thêm biến thể từ ảnh gốc để mô hình nhận diện
tốt hơn trong điều kiện thực tế khác với ảnh training.

8 biến thể được tạo cho mỗi ảnh gốc:
  1. blur_mild       — mờ nhẹ (nhân tạo điều kiện lens không nét)
  2. blur_heavy      — mờ nặng (chuyển động mạnh)
  3. motion_blur     — mờ chuyển động ngang
  4. low_light_mild  — tối nhẹ (ánh sáng yếu)
  5. low_light_heavy — tối nặng (ban đêm, bóng tối)
  6. noise           — nhiễu ngẫu nhiên (camera chất lượng thấp)
  7. low_quality     — JPEG compression artifact
  8. combined        — kết hợp nhiều hiệu ứng cùng lúc

Khi nào nên dùng:
  ✓ Có ÍT ảnh gốc (< 20 ảnh/người)
  ✓ Môi trường nhận diện có ánh sáng thay đổi
  ✓ Muốn cải thiện độ chính xác mà không quay thêm video

Khi KHÔNG nên dùng:
  ✗ Đã có > 50 ảnh gốc/người (augment sẽ tạo quá nhiều file)
  ✗ Sau khi augment, nhớ chạy "Rebuild encodings" để cập nhật

File augment được đặt tên: <tên>_aug_<biến thể>.jpg
PersonManager đếm riêng ảnh gốc và ảnh augment.
""")


def _help_rebuild() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   REBUILD ENCODINGS                          ║
╚══════════════════════════════════════════════════════════════╝

Rebuild đọc lại toàn bộ ảnh trong dataset và tạo mới file
face_encodings_hybrid.pkl — đây là file mà recognizer.py dùng
để nhận diện khuôn mặt.

Khi nào BẮT BUỘC phải rebuild:
  ✓ Vừa thêm ảnh mới (thu thập / augment)
  ✓ Vừa đổi model YOLO (ví dụ: yolov11n → yolov12s)
  ✓ Vừa xóa / reset một person
  ✓ File pkl bị lỗi hoặc mất

Rebuild vs Optimize:
  Rebuild  — tạo lại từ đầu, đọc toàn bộ ảnh (~chậm)
  Optimize — gộp encoding giống nhau trong pkl đã có (~nhanh)
  → Thứ tự đúng: Rebuild trước → Optimize sau (tùy chọn)

Thời gian rebuild phụ thuộc vào:
  • Số lượng ảnh trong dataset
  • Model YOLO (n nhanh hơn l)
  • Tốc độ CPU

File output: model/face_encodings_hybrid.pkl
File cũ được backup: model/face_encodings_hybrid_backup.pkl
""")


def _help_optimize() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  OPTIMIZE ENCODINGS                          ║
╚══════════════════════════════════════════════════════════════╝

Optimize gộp các encoding quá giống nhau (clustering) và giữ
tối đa MAX_ENCODINGS_PER_PERSON (mặc định 30) encoding/người.

Tại sao cần optimize:
  • Nhiều ảnh augment → hàng trăm encoding → nhận diện chậm
  • Encoding trùng lặp không tăng độ chính xác
  • Optimize giữ lại encoding đa dạng nhất

Thuật toán:
  1. Tính khoảng cách giữa tất cả encoding (pairwise distance)
  2. Gộp encoding có distance < CLUSTERING_THRESHOLD (0.15)
  3. Chọn encoding gần centroid nhất từ mỗi cụm
  4. Nếu vẫn > 30: lấy đều từ dải chất lượng thấp → cao

Kết quả điển hình:
  Trước optimize: 200-400 encodings → Sau: 50-90 encodings
  Tốc độ nhận diện tăng ~3-5x

Khi nào chạy:
  → Sau Rebuild, nếu nhận diện cảm thấy chậm
  → Sau Augment + Rebuild (augment tạo rất nhiều encoding)

Lưu ý: Optimize đọc từ pkl hiện có, không cần đọc lại ảnh.
""")


def _help_settings() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║             CÀI ĐẶT THU THẬP FRAME — Ý NGHĨA THAM SỐ       ║
╚══════════════════════════════════════════════════════════════╝

min_laplacian  [mặc định: 15]
  Độ nét tối thiểu (Laplacian variance).
  Thấp hơn → chấp nhận ảnh mờ hơn.
  Video mp4v nén: thường 6-43 (thấp hơn ảnh raw).
  Tăng lên nếu muốn chỉ lấy ảnh thật sắc nét (> 30).

min_face_px  [mặc định: 70]
  Kích thước khuôn mặt tối thiểu (pixel — cạnh nhỏ hơn).
  Thấp hơn → chấp nhận khuôn mặt nhỏ (đứng xa hơn).
  Giảm về 50 nếu người hay đứng xa camera.

detect_conf  [mặc định: 0.15]
  Ngưỡng confidence tối thiểu của YOLO để nhận diện face.
  Thấp hơn → bắt được khuôn mặt nghiêng / xa / mờ.
  Tăng lên 0.4+ nếu gặp false detection (vật thể bị nhận là mặt).

ssim_threshold  [mặc định: 0.85]
  Ngưỡng similarity giữa frame mới và frame vừa lưu.
  Cao hơn → khó bị bỏ qua (lưu nhiều frame giống nhau hơn).
  Thấp hơn → khắt khe hơn về độ đa dạng.
  Khoảng: 0.7 (rất đa dạng) — 0.95 (chấp nhận giống nhau nhiều)

min_frame_gap  [mặc định: 10]
  Số frame tối thiểu giữa 2 lần lưu liên tiếp.
  Tăng lên nếu video nhiều frame giống nhau (video tĩnh).
  Giảm xuống 3-5 nếu video ngắn mà muốn nhiều frame.

frame_skip  [mặc định: 3]
  Bỏ qua bao nhiêu frame trước khi xử lý 1 frame.
  1 = xử lý mọi frame (chậm nhưng đầy đủ nhất).
  5 = bỏ 4 frame, xử lý 1 (nhanh hơn 5x).

Presets có sẵn:
  Thoai mai — bắt nhiều frame, it reject nhất
  Binh thuong — khuyến nghị (mặc định)
  Khat khe — chỉ lấy frame chất lượng cao
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli = InteractiveCLI()
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
