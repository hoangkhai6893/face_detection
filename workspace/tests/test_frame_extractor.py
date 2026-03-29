"""
Unit tests for FrameDiversityFilter and VideoFrameExtractor.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from core.frame_extractor import (
    ExtractedFrame,
    FrameDiversityFilter,
    FrameQualityChecker,
    VideoFrameExtractor,
)


# ---------------------------------------------------------------------------
# FrameDiversityFilter
# ---------------------------------------------------------------------------

class TestFrameDiversityFilter:
    @pytest.fixture
    def filt(self):
        return FrameDiversityFilter(ssim_threshold=0.85, min_frame_gap=15)

    def test_first_frame_always_accepted(self, filt, synthetic_face_image):
        assert filt.is_diverse(synthetic_face_image, 0)

    def test_identical_frame_rejected(self, filt, synthetic_face_image):
        filt.accept(synthetic_face_image, 0)
        # Same image again, but gap is large
        result = filt.is_diverse(synthetic_face_image, 100)
        assert not result, "Identical frame should be rejected (SSIM ≈ 1.0)"

    def test_different_frame_accepted(self, filt, synthetic_face_image, dark_image):
        filt.accept(synthetic_face_image, 0)
        # dark_image is visually very different
        assert filt.is_diverse(dark_image, 100)

    def test_min_frame_gap_enforced(self, filt, synthetic_face_image, dark_image):
        filt.accept(synthetic_face_image, 0)
        # Frame 5 is too soon (gap=5 < min_frame_gap=15)
        assert not filt.is_diverse(dark_image, 5), \
            "Should reject frame too close in time regardless of content"

    def test_ssim_threshold_configurable(self, synthetic_face_image):
        strict = FrameDiversityFilter(ssim_threshold=0.1, min_frame_gap=1)
        strict.accept(synthetic_face_image, 0)
        # Even slightly different image should be accepted with strict threshold
        slightly_different = synthetic_face_image.copy()
        slightly_different[0, 0] = [0, 0, 0]
        # With threshold 0.1, even a very similar image might be rejected.
        # Here we test that the filter works — SSIM of near-identical images
        # should be > 0.1, so the frame IS rejected (not diverse enough).
        result = strict.is_diverse(slightly_different, 100)
        assert not result, "Strict SSIM threshold should reject near-identical frames"

    def test_reset_clears_state(self, filt, synthetic_face_image, dark_image):
        filt.accept(synthetic_face_image, 0)
        filt.reset()
        # After reset, even the same image should be accepted (no prior frame)
        assert filt.is_diverse(synthetic_face_image, 1)


# ---------------------------------------------------------------------------
# VideoFrameExtractor (with mocked YOLO)
# ---------------------------------------------------------------------------

def _make_mock_yolo(face_image: np.ndarray):
    """Return a YOLO mock that always detects one face covering most of the image."""
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


def _make_no_face_yolo():
    """Return a YOLO mock that never detects a face."""
    result = MagicMock()
    result.boxes = []
    yolo = MagicMock()
    yolo.return_value = [result]
    return yolo


class TestVideoFrameExtractor:
    def test_extract_yields_frames_from_valid_video(self, synthetic_video, synthetic_face_image):
        yolo = _make_mock_yolo(synthetic_face_image)
        extractor = VideoFrameExtractor(
            yolo_model=yolo,
            quality_checker=FrameQualityChecker(min_laplacian=0.0, min_face_size=10),
            diversity_filter=FrameDiversityFilter(ssim_threshold=0.99, min_frame_gap=1),
            frame_skip=1,
        )
        frames = list(extractor.extract_from_file(str(synthetic_video), max_frames=5))
        assert len(frames) >= 1, "Should extract at least one frame from valid video"

    def test_extract_returns_extracted_frame_type(self, synthetic_video, synthetic_face_image):
        yolo = _make_mock_yolo(synthetic_face_image)
        extractor = VideoFrameExtractor(
            yolo_model=yolo,
            quality_checker=FrameQualityChecker(min_laplacian=0.0, min_face_size=10),
            diversity_filter=FrameDiversityFilter(ssim_threshold=0.99, min_frame_gap=1),
            frame_skip=1,
        )
        frames = list(extractor.extract_from_file(str(synthetic_video), max_frames=1))
        if frames:
            ef = frames[0]
            assert isinstance(ef, ExtractedFrame)
            assert isinstance(ef.face_crop, np.ndarray)
            assert 0.0 <= ef.quality_score <= 1.0

    def test_extract_respects_max_frames(self, synthetic_video, synthetic_face_image):
        yolo = _make_mock_yolo(synthetic_face_image)
        extractor = VideoFrameExtractor(
            yolo_model=yolo,
            quality_checker=FrameQualityChecker(min_laplacian=0.0, min_face_size=10),
            diversity_filter=FrameDiversityFilter(ssim_threshold=0.99, min_frame_gap=1),
            frame_skip=1,
        )
        frames = list(extractor.extract_from_file(str(synthetic_video), max_frames=3))
        assert len(frames) <= 3

    def test_no_face_yields_no_frames(self, synthetic_video):
        """When YOLO returns no boxes and HOG also fails, no frames should be yielded."""
        yolo = _make_no_face_yolo()
        extractor = VideoFrameExtractor(yolo_model=yolo, frame_skip=1)

        # Also patch HOG fallback so it returns nothing
        with patch.object(extractor, "_detect_face_hog", return_value=(None, None, 0.0)):
            frames = list(extractor.extract_from_file(str(synthetic_video), max_frames=10))
        assert frames == [], "No frames should be saved when no face is detected"

    def test_blurry_frames_not_saved(self, blurry_video, blurry_image):
        """Blurry frames should fail the quality gate."""
        yolo = _make_mock_yolo(blurry_image)
        extractor = VideoFrameExtractor(
            yolo_model=yolo,
            # Default min_laplacian=100 — blurry_image will fail
            quality_checker=FrameQualityChecker(min_laplacian=100.0, min_face_size=10),
            frame_skip=1,
        )
        with patch.object(extractor, "_detect_face_hog", return_value=(None, None, 0.0)):
            frames = list(extractor.extract_from_file(str(blurry_video), max_frames=10))
        assert frames == [], "Blurry frames should be rejected"

    def test_nonexistent_video_yields_nothing(self):
        yolo = MagicMock()
        extractor = VideoFrameExtractor(yolo_model=yolo)
        frames = list(extractor.extract_from_file("/nonexistent/path.mp4", max_frames=5))
        assert frames == []
