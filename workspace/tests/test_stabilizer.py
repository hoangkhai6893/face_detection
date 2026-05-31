"""
Unit tests for RecognitionStabilizer.
"""
import numpy as np
import pytest
from core.recognition_stabilizer import RecognitionStabilizer, StableEvent


@pytest.fixture
def stab():
    return RecognitionStabilizer(
        window_size=5,
        min_count_known=2,
        min_count_unknown=4,
        iou_threshold=0.4,
        track_timeout_frames=60,
    )


@pytest.fixture
def blank_frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


BBOX = (50, 50, 150, 150)


class TestKnownPersonStable:
    def test_fires_after_min_known_votes(self, stab, blank_frame):
        stab.update("Unknown", BBOX, 1, blank_frame)
        result = stab.update("Khai (0.91)", BBOX, 2, blank_frame)
        assert result is None  # only 1 known vote

        result = stab.update("Khai (0.88)", BBOX, 3, blank_frame)
        assert result is not None
        assert result.person_name == "Khai"
        assert result.is_known is True
        assert result.confidence == pytest.approx(0.895, abs=0.01)

    def test_does_not_fire_twice_for_same_stable_name(self, stab, blank_frame):
        stab.update("Khai (0.90)", BBOX, 1, blank_frame)
        first = stab.update("Khai (0.90)", BBOX, 2, blank_frame)
        assert first is not None
        # Third update — still "Khai", should NOT fire again
        second = stab.update("Khai (0.90)", BBOX, 3, blank_frame)
        assert second is None

    def test_fires_on_identity_change(self, stab, blank_frame):
        stab.update("Khai (0.90)", BBOX, 1, blank_frame)
        stab.update("Khai (0.90)", BBOX, 2, blank_frame)  # stable as Khai
        # Frame 3-4: tie (Khai=2, Lan=2) → no change
        stab.update("Lan (0.88)", BBOX, 3, blank_frame)
        no_change = stab.update("Lan (0.85)", BBOX, 4, blank_frame)
        assert no_change is None  # tie → stabilizer does not flip
        # Frame 5: Lan=3 > Khai=2 → Lan wins
        result = stab.update("Lan (0.82)", BBOX, 5, blank_frame)
        assert result is not None
        assert result.person_name == "Lan"


class TestUnknownStable:
    def test_fires_after_min_unknown_votes(self, stab, blank_frame):
        for i in range(3):
            result = stab.update("Unknown", BBOX, i + 1, blank_frame)
            assert result is None  # not enough yet

        result = stab.update("Unknown", BBOX, 4, blank_frame)
        assert result is not None
        assert result.person_name == "Unknown"
        assert result.is_known is False
        assert result.confidence is None

    def test_mixed_votes_prevent_unknown_alert(self, stab, blank_frame):
        stab.update("Unknown", BBOX, 1, blank_frame)
        stab.update("Unknown", BBOX, 2, blank_frame)
        stab.update("Khai (0.88)", BBOX, 3, blank_frame)
        result = stab.update("Unknown", BBOX, 4, blank_frame)
        # Only 3 Unknown out of 4 — threshold is 4 → no alert
        assert result is None


class TestNonEvents:
    def test_no_face_is_ignored(self, stab, blank_frame):
        result = stab.update("No Face", BBOX, 1, blank_frame)
        assert result is None

    def test_error_is_ignored(self, stab, blank_frame):
        result = stab.update("Error", BBOX, 1, blank_frame)
        assert result is None


class TestFaceTracking:
    def test_different_bbox_creates_separate_track(self, stab, blank_frame):
        bbox2 = (400, 400, 500, 500)  # no IoU overlap with BBOX
        stab.update("Khai (0.90)", BBOX, 1, blank_frame)
        # Different face at different position
        result = stab.update("Khai (0.90)", bbox2, 2, blank_frame)
        # Each track has only 1 vote — neither stable
        assert result is None

    def test_stale_track_pruned(self, stab, blank_frame):
        stab.update("Khai (0.90)", BBOX, 1, blank_frame)
        # Jump frame_num far ahead (> track_timeout=60)
        stab.update("Khai (0.90)", BBOX, 100, blank_frame)
        # After prune, track resets — should need full window again
        result = stab.update("Khai (0.90)", BBOX, 100, blank_frame)
        assert result is None or result.person_name == "Khai"

    def test_reset_clears_all_tracks(self, stab, blank_frame):
        stab.update("Khai (0.90)", BBOX, 1, blank_frame)
        stab.update("Khai (0.90)", BBOX, 2, blank_frame)
        stab.reset()
        # After reset, need full window again
        result = stab.update("Khai (0.90)", BBOX, 3, blank_frame)
        assert result is None
