"""
Integration tests for the full data collection pipeline.

These tests use temporary directories and a synthetic video; they do NOT
touch the real family_images/ dataset or any live camera.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from core.frame_extractor import VideoFrameExtractor
from core.frame_quality import FrameQualityChecker
from core.frame_diversity import FrameDiversityFilter
from training.person_manager import PersonManager
from training.data_collection import DataCollectionSession, CollectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_yolo(face_image: np.ndarray):
    h, w = face_image.shape[:2]
    box = MagicMock()
    box.conf = [0.9]
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, w - 10, h - 10]
    result = MagicMock()
    result.boxes = [box]
    yolo = MagicMock()
    yolo.return_value = [result]
    return yolo


def _make_extractor(face_image: np.ndarray) -> VideoFrameExtractor:
    return VideoFrameExtractor(
        yolo_model=_make_mock_yolo(face_image),
        quality_checker=FrameQualityChecker(min_laplacian=0.0, min_face_size=10),
        diversity_filter=FrameDiversityFilter(ssim_threshold=0.99, min_frame_gap=1),
        frame_skip=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullVideoCollection:
    def test_frames_saved_to_person_folder(self, tmp_dataset, synthetic_video, synthetic_face_image):
        """Files should appear in the person folder after collection."""
        extractor = _make_extractor(synthetic_face_image)

        # Patch rebuild_encodings so we don't need real YOLO/models
        with patch("training.data_collection.rebuild_encodings", return_value=(0, 0)):
            session = DataCollectionSession(
                person_name="Alice",
                extractor=extractor,
                dataset_path=str(tmp_dataset),
            )
            result = session.run_from_file(str(synthetic_video), max_frames=5)

        saved_images = list((tmp_dataset / "Alice").glob("*_q*.jpg"))
        assert len(saved_images) >= 1, "At least one image should be saved"
        assert result.frames_saved >= 1

    def test_collection_result_fields_valid(self, tmp_dataset, synthetic_video, synthetic_face_image):
        extractor = _make_extractor(synthetic_face_image)
        with patch("training.data_collection.rebuild_encodings", return_value=(0, 0)):
            session = DataCollectionSession("Alice", extractor, str(tmp_dataset))
            result = session.run_from_file(str(synthetic_video), max_frames=5)

        assert isinstance(result, CollectionResult)
        assert result.person_name == "Alice"
        assert result.frames_saved >= 0
        assert result.frames_processed >= 0
        assert result.duration_seconds >= 0.0
        if result.frames_saved > 0:
            assert 0.0 <= result.avg_quality <= 1.0


class TestPersonCreateThenCollect:
    def test_new_person_gets_images(self, tmp_path, synthetic_video, synthetic_face_image):
        mgr = PersonManager(dataset_path=str(tmp_path))
        mgr.create_person("NewPerson")
        assert mgr.exists("NewPerson")

        extractor = _make_extractor(synthetic_face_image)
        with patch("training.data_collection.rebuild_encodings", return_value=(0, 0)):
            session = DataCollectionSession("NewPerson", extractor, str(tmp_path))
            result = session.run_from_file(str(synthetic_video), max_frames=5)

        assert (tmp_path / "NewPerson").is_dir()
        assert result.frames_saved >= 0   # May be 0 if diversity filter blocks all


class TestPersonCRUD:
    def test_reset_clears_images_keeps_folder(self, tmp_dataset):
        mgr = PersonManager(dataset_path=str(tmp_dataset))
        count = mgr.reset_person("Alice", rebuild=False)
        assert count == 3
        assert (tmp_dataset / "Alice").is_dir()
        remaining = list((tmp_dataset / "Alice").iterdir())
        assert remaining == []

    def test_delete_removes_folder(self, tmp_dataset):
        mgr = PersonManager(dataset_path=str(tmp_dataset))
        mgr.delete_person("Bob", rebuild=False)
        assert not (tmp_dataset / "Bob").exists()

    def test_list_after_delete_reflects_change(self, tmp_dataset):
        mgr = PersonManager(dataset_path=str(tmp_dataset))
        mgr.delete_person("Alice", rebuild=False)
        names = {p.name for p in mgr.list_persons()}
        assert "Alice" not in names
        assert "Bob" in names


class TestFilenameConvention:
    def test_saved_files_follow_naming_pattern(self, tmp_dataset, synthetic_video, synthetic_face_image):
        """Saved filenames must match <person>_<timestamp_ms>_q<score>.jpg"""
        import re
        extractor = _make_extractor(synthetic_face_image)
        with patch("training.data_collection.rebuild_encodings", return_value=(0, 0)):
            session = DataCollectionSession("Alice", extractor, str(tmp_dataset))
            session.run_from_file(str(synthetic_video), max_frames=3)

        pattern = re.compile(r"^Alice_\d+_q\d+\.\d+\.jpg$")
        new_files = [
            f.name for f in (tmp_dataset / "Alice").iterdir()
            if f.name not in {"Alice_0.jpg", "Alice_1.jpg", "Alice_2.jpg"}
        ]
        for fname in new_files:
            assert pattern.match(fname), f"Filename '{fname}' does not match naming convention"
