"""
Unit tests for MotionGuard state machine.

Tests focus on the logic layer (state transitions, probe burst, counters),
not on pixel-level motion detection which depends on OpenCV internals.
_detect_motion() is mocked via monkeypatch where predictable behaviour is needed.
"""

import time
import logging
import numpy as np
import pytest

from core.motion_guard import MotionGuard, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLACK_FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _make_guard(
    idle_sample_every=1,
    no_face_limit=3,
    absdiff_threshold=8,
    mog2_threshold=5,
    max_idle_sec=999,
    probe_burst=3,
) -> MotionGuard:
    return MotionGuard(
        idle_sample_every=idle_sample_every,
        no_face_limit=no_face_limit,
        absdiff_threshold=absdiff_threshold,
        mog2_threshold=mog2_threshold,
        max_idle_sec=max_idle_sec,
        probe_burst=probe_burst,
        logger=logging.getLogger("test"),
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_idle(self):
        guard = _make_guard()
        assert guard.state == State.IDLE

    def test_is_not_active_initially(self):
        guard = _make_guard()
        assert guard.is_active is False

    def test_probe_count_zero_on_init(self):
        guard = _make_guard()
        assert guard._probe_count == 0


# ---------------------------------------------------------------------------
# IDLE → ACTIVE via motion
# ---------------------------------------------------------------------------

class TestIdleToActive:
    def test_motion_detected_transitions_active(self, monkeypatch):
        guard = _make_guard()
        monkeypatch.setattr(guard, "_detect_motion", lambda f: True)
        result = guard.should_process(BLACK_FRAME)
        assert result is True
        assert guard.state == State.ACTIVE

    def test_no_motion_stays_idle(self, monkeypatch):
        guard = _make_guard()
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        result = guard.should_process(BLACK_FRAME)
        assert result is False
        assert guard.state == State.IDLE

    def test_should_process_true_on_active_entry(self, monkeypatch):
        guard = _make_guard()
        monkeypatch.setattr(guard, "_detect_motion", lambda f: True)
        assert guard.should_process(BLACK_FRAME) is True


# ---------------------------------------------------------------------------
# ACTIVE state
# ---------------------------------------------------------------------------

class TestActiveState:
    def test_active_always_returns_true(self):
        guard = _make_guard()
        guard._state = State.ACTIVE   # force into ACTIVE
        for _ in range(5):
            assert guard.should_process(BLACK_FRAME) is True

    def test_active_is_true(self):
        guard = _make_guard()
        guard._state = State.ACTIVE
        assert guard.is_active is True


# ---------------------------------------------------------------------------
# ACTIVE → IDLE via report_faces
# ---------------------------------------------------------------------------

class TestActiveToIdle:
    def test_no_face_limit_transitions_to_idle(self):
        guard = _make_guard(no_face_limit=3)
        guard._state = State.ACTIVE

        for _ in range(3):
            guard.report_faces(0)

        assert guard.state == State.IDLE

    def test_face_resets_no_face_counter(self):
        guard = _make_guard(no_face_limit=3)
        guard._state = State.ACTIVE

        guard.report_faces(0)
        guard.report_faces(0)
        assert guard._no_face_count == 2

        guard.report_faces(1)   # face → reset
        assert guard._no_face_count == 0
        assert guard.state == State.ACTIVE  # still ACTIVE

    def test_partial_no_face_then_face_stays_active(self):
        guard = _make_guard(no_face_limit=5)
        guard._state = State.ACTIVE

        for _ in range(4):
            guard.report_faces(0)

        guard.report_faces(1)   # face before limit → stays ACTIVE

        for _ in range(4):
            guard.report_faces(0)

        assert guard.state == State.ACTIVE  # counter reset; not at limit yet

    def test_multiple_faces_counted_as_one_event(self):
        guard = _make_guard(no_face_limit=3)
        guard._state = State.ACTIVE
        guard.report_faces(5)   # 5 faces: still just 1 reset
        assert guard._no_face_count == 0

    def test_idle_skip_resets_on_idle_transition(self):
        guard = _make_guard(no_face_limit=2)
        guard._state = State.ACTIVE
        guard._idle_skip = 4

        guard.report_faces(0)
        guard.report_faces(0)   # → IDLE

        assert guard.state == State.IDLE
        assert guard._idle_skip == 0


# ---------------------------------------------------------------------------
# report_faces when IDLE (probe logic)
# ---------------------------------------------------------------------------

class TestReportFacesIdle:
    def test_idle_no_face_does_nothing(self):
        guard = _make_guard()
        guard._state = State.IDLE
        guard.report_faces(0)
        assert guard.state == State.IDLE

    def test_idle_face_during_probe_transitions_active(self):
        guard = _make_guard()
        guard._state = State.IDLE
        guard._probe_count = 2   # simulate probe in progress

        guard.report_faces(1)    # face found → ACTIVE
        assert guard.state == State.ACTIVE
        assert guard._probe_count == 0  # cleared

    def test_idle_face_clears_probe_count(self):
        guard = _make_guard()
        guard._state = State.IDLE
        guard._probe_count = 5
        guard.report_faces(1)
        assert guard._probe_count == 0


# ---------------------------------------------------------------------------
# Probe burst
# ---------------------------------------------------------------------------

class TestProbeBurst:
    def test_probe_burst_runs_without_state_change(self, monkeypatch):
        """Probe: frames run YOLO but state stays IDLE as long as no face reported."""
        guard = _make_guard(max_idle_sec=0, probe_burst=3)
        guard._state = State.IDLE
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        guard._last_yolo_time = time.time() - 1  # force timeout

        # First call: triggers probe burst (probe_burst - 1 remaining after this)
        r1 = guard.should_process(BLACK_FRAME)
        assert r1 is True
        assert guard.state == State.IDLE   # NOT ACTIVE — probe doesn't change state
        assert guard._probe_count == 2     # 3 total, this frame used 1

        # Second and third: consume remaining probe frames
        r2 = guard.should_process(BLACK_FRAME)
        assert r2 is True
        assert guard.state == State.IDLE

        r3 = guard.should_process(BLACK_FRAME)
        assert r3 is True
        assert guard.state == State.IDLE

    def test_probe_ends_after_burst(self, monkeypatch):
        """After probe_burst frames, probe_count returns to 0."""
        # Use a large max_idle_sec so the timeout does NOT re-trigger during the test.
        guard = _make_guard(max_idle_sec=9999, probe_burst=3)
        guard._state = State.IDLE
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        # Force exactly one timeout by setting last_yolo_time far in the past once.
        guard._last_yolo_time = time.time() - 10000

        # First call: timeout triggers, probe starts (probe_count = probe_burst - 1 = 2)
        guard.should_process(BLACK_FRAME)
        assert guard._probe_count == 2

        # Consume remaining probe frames
        guard.should_process(BLACK_FRAME)  # probe_count → 1
        guard.should_process(BLACK_FRAME)  # probe_count → 0

        # Probe exhausted
        assert guard._probe_count == 0

        # Next call: no motion, and timeout won't re-trigger yet (last_yolo_time was
        # reset at probe start, and max_idle_sec=9999 → far future)
        r = guard.should_process(BLACK_FRAME)
        assert r is False

    def test_probe_face_triggers_active(self, monkeypatch):
        guard = _make_guard(max_idle_sec=0, probe_burst=3)
        guard._state = State.IDLE
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        guard._last_yolo_time = time.time() - 1

        guard.should_process(BLACK_FRAME)   # probe starts
        assert guard.state == State.IDLE

        guard.report_faces(1)               # face found during probe
        assert guard.state == State.ACTIVE

    def test_probe_no_face_stays_idle(self, monkeypatch):
        guard = _make_guard(max_idle_sec=0, probe_burst=3)
        guard._state = State.IDLE
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        guard._last_yolo_time = time.time() - 1

        guard.should_process(BLACK_FRAME)
        guard.report_faces(0)               # no face → stays IDLE
        assert guard.state == State.IDLE

    def test_probe_resets_last_yolo_time(self, monkeypatch):
        guard = _make_guard(max_idle_sec=0, probe_burst=3)
        guard._state = State.IDLE
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)
        old_time = time.time() - 10
        guard._last_yolo_time = old_time

        guard.should_process(BLACK_FRAME)

        # Timer must have been reset (new value > old)
        assert guard._last_yolo_time > old_time


# ---------------------------------------------------------------------------
# IDLE throttle (idle_sample_every)
# ---------------------------------------------------------------------------

class TestIdleThrottle:
    def test_idle_skips_non_sample_frames(self, monkeypatch):
        """With idle_sample_every=3, only every 3rd frame runs motion check."""
        guard = _make_guard(idle_sample_every=3, max_idle_sec=999)
        monkeypatch.setattr(guard, "_detect_motion", lambda f: False)

        # Frame 1: _idle_skip → 1, check 1<3 → True → skip (return False)
        assert guard.should_process(BLACK_FRAME) is False
        # Frame 2: _idle_skip → 2 < 3 → skip
        assert guard.should_process(BLACK_FRAME) is False
        # Frame 3: _idle_skip → 3, 3 < 3 = False → reset, run motion (False) → False
        assert guard.should_process(BLACK_FRAME) is False

    def test_idle_runs_on_sample_frame(self, monkeypatch):
        guard = _make_guard(idle_sample_every=2, max_idle_sec=999)
        motion_called = []
        monkeypatch.setattr(guard, "_detect_motion", lambda f: motion_called.append(1) or False)

        guard.should_process(BLACK_FRAME)  # frame 1: skip
        assert len(motion_called) == 0

        guard.should_process(BLACK_FRAME)  # frame 2: runs
        assert len(motion_called) == 1


# ---------------------------------------------------------------------------
# Idiomatic full cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    def test_idle_active_idle_cycle(self, monkeypatch):
        guard = _make_guard(no_face_limit=3)
        monkeypatch.setattr(guard, "_detect_motion", lambda f: True)

        # IDLE → ACTIVE
        guard.should_process(BLACK_FRAME)
        guard.report_faces(2)
        assert guard.state == State.ACTIVE

        # ACTIVE → IDLE
        for _ in range(3):
            guard.report_faces(0)
        assert guard.state == State.IDLE

        # IDLE → ACTIVE again
        guard.should_process(BLACK_FRAME)
        guard.report_faces(1)
        assert guard.state == State.ACTIVE
