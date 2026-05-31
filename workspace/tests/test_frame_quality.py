"""
Unit tests for FrameQualityChecker.
"""

import numpy as np
import pytest

from core.frame_quality import FrameQualityChecker


@pytest.fixture
def checker():
    return FrameQualityChecker(
        min_laplacian=100.0,
        min_brightness=0.10,
        max_brightness=0.92,
        min_face_size=100,
    )


class TestSharpness:
    def test_sharp_image_passes(self, checker, synthetic_face_image):
        assert checker.is_sharp(synthetic_face_image), \
            "Sharp synthetic image should pass sharpness gate"

    def test_blurry_image_fails(self, checker, blurry_image):
        assert not checker.is_sharp(blurry_image), \
            "Heavily blurred image should fail sharpness gate"


class TestBrightness:
    def test_normal_brightness_passes(self, checker, synthetic_face_image):
        assert checker.is_bright(synthetic_face_image), \
            "Moderately bright image should pass brightness gate"

    def test_dark_image_fails(self, checker, dark_image):
        assert not checker.is_bright(dark_image), \
            "Very dark image (brightness < 0.10) should fail"

    def test_overexposed_image_fails(self, checker):
        # Create a pure near-white image (brightness ≈ 0.98) guaranteed to fail
        white = np.full((200, 200, 3), 250, dtype=np.uint8)
        assert not checker.is_bright(white), \
            "Near-white image (brightness > 0.92) should fail"


class TestSize:
    def test_adequate_size_passes(self, checker, synthetic_face_image):
        assert checker.is_large_enough(synthetic_face_image), \
            "200×200 image should pass size gate (min=100)"

    def test_tiny_face_fails(self, checker, tiny_face_image):
        assert not checker.is_large_enough(tiny_face_image), \
            "50×50 image should fail size gate (min=100)"


class TestQualityScore:
    def test_score_in_valid_range(self, checker, synthetic_face_image):
        s = checker.score(synthetic_face_image)
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1]"

    def test_sharp_scores_higher_than_blurry(self, checker, synthetic_face_image, blurry_image):
        assert checker.score(synthetic_face_image) > checker.score(blurry_image)

    def test_passes_all_good_image(self, checker, synthetic_face_image):
        assert checker.passes_all(synthetic_face_image)

    def test_reject_reason_blurry(self, checker, blurry_image):
        reason = checker.reject_reason(blurry_image)
        assert reason == "blurry"

    def test_reject_reason_dark(self, checker, dark_image):
        reason = checker.reject_reason(dark_image)
        assert reason == "too_dark"

    def test_reject_reason_small(self, checker, tiny_face_image):
        reason = checker.reject_reason(tiny_face_image)
        assert reason == "too_small"

    def test_no_reject_reason_for_good_image(self, checker, synthetic_face_image):
        assert checker.reject_reason(synthetic_face_image) is None
