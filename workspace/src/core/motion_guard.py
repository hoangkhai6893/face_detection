#!/usr/bin/env python3
"""
MotionGuard — state machine IDLE/ACTIVE cho camera-based face recognition.

             motion detected
  ┌────────┐ ──────────────► ┌────────┐
  │  IDLE  │                 │ ACTIVE │
  │ ~2%CPU │ ◄────────────── │ ~50%CPU│
  └────────┘  no face N sec  └────────┘

IDLE:
  - Sample every IDLE_SAMPLE_EVERY frames (~5fps khi camera chạy 30fps)
  - Chỉ chạy absdiff + MOG2 trên ảnh 160×90 (~0.6ms)
  - Khi phát hiện motion → chuyển sang ACTIVE

ACTIVE:
  - Chạy full pipeline (YOLO + dlib) mỗi frame (có thể kết hợp với motion gate nhẹ)
  - Khi YOLO liên tục trả về 0 mặt trong IDLE_NO_FACE_FRAMES frames → về IDLE

Usage:
    guard = MotionGuard()

    while camera:
        frame = cap.read()
        if guard.should_process(frame):
            detections = yolo.detect(frame)
            guard.report_faces(len(detections))
            # ... recognition ...
        else:
            # reuse last result
"""

from __future__ import annotations

import logging
import sys
import os
import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class State(Enum):
    IDLE   = auto()
    ACTIVE = auto()


class MotionGuard:
    # Resolution dùng cho motion detection — nhỏ để tiết kiệm CPU
    _MOTION_W = 160
    _MOTION_H = 90

    def __init__(
        self,
        idle_sample_every: int   = config.IDLE_SAMPLE_EVERY,
        no_face_limit: int        = config.IDLE_NO_FACE_FRAMES,
        absdiff_threshold: float  = config.MOTION_ABSDIFF_THRESHOLD,
        mog2_threshold: float     = config.MOTION_MOG2_THRESHOLD,
        max_idle_sec: float       = config.MOTION_MAX_IDLE_SEC,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._idle_sample_every = idle_sample_every
        self._no_face_limit     = no_face_limit
        self._absdiff_threshold = absdiff_threshold
        self._mog2_threshold    = mog2_threshold
        self._max_idle_sec      = max_idle_sec
        self._logger = logger or logging.getLogger(__name__)

        # State machine
        self._state          = State.IDLE
        self._idle_skip      = 0       # frame counter để throttle khi IDLE
        self._no_face_count  = 0       # frame ACTIVE liên tiếp không có mặt
        self._last_yolo_time = time.time()

        # Motion detection internals
        self._prev_gray: Optional[np.ndarray] = None
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == State.ACTIVE

    def should_process(self, frame: np.ndarray) -> bool:
        """
        Gọi mỗi frame. Trả về True khi nên chạy full pipeline (YOLO + dlib).

        IDLE  → chỉ check motion mỗi IDLE_SAMPLE_EVERY frame.
        ACTIVE → luôn True (+ idle fallback nếu pipeline không chạy quá lâu).
        """
        if self._state == State.IDLE:
            return self._idle_step(frame)
        else:
            return self._active_step(frame)

    def report_faces(self, face_count: int) -> None:
        """
        Gọi sau mỗi lần YOLO chạy với số mặt phát hiện được.
        Dùng để quyết định có về IDLE không.
        """
        self._last_yolo_time = time.time()

        if self._state != State.ACTIVE:
            return

        if face_count > 0:
            self._no_face_count = 0
        else:
            self._no_face_count += 1
            if self._no_face_count >= self._no_face_limit:
                self._state = State.IDLE
                self._idle_skip = 0
                self._no_face_count = 0
                self._logger.info(
                    "MotionGuard: ACTIVE → IDLE (no face for %d frames)",
                    self._no_face_limit,
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _idle_step(self, frame: np.ndarray) -> bool:
        """Throttle: chỉ check motion mỗi N frame khi IDLE."""
        self._idle_skip += 1
        if self._idle_skip < self._idle_sample_every:
            return False
        self._idle_skip = 0

        if self._detect_motion(frame):
            self._state = State.ACTIVE
            self._no_face_count = 0
            self._logger.info("MotionGuard: IDLE → ACTIVE (motion detected)")
            return True

        # Idle fallback: force-run nếu quá lâu không xử lý
        # (tránh người đứng yên ngay từ đầu không bị bỏ qua)
        if (time.time() - self._last_yolo_time) > self._max_idle_sec:
            self._state = State.ACTIVE
            self._no_face_count = 0
            self._logger.info("MotionGuard: IDLE → ACTIVE (idle timeout)")
            return True

        return False

    def _active_step(self, frame: np.ndarray) -> bool:
        """Trong ACTIVE: luôn process. Không cần check motion."""
        # (report_faces() xử lý việc chuyển về IDLE)
        return True

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """absdiff + MOG2 trên ảnh 160×90. ~0.6ms tổng."""
        small = cv2.resize(frame, (self._MOTION_W, self._MOTION_H))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # absdiff: kiểm tra nhanh có gì thay đổi không
        if self._prev_gray is not None:
            absdiff_score = float(cv2.absdiff(self._prev_gray, gray).mean())
        else:
            absdiff_score = 255.0   # frame đầu tiên: luôn trigger
        self._prev_gray = gray

        if absdiff_score < self._absdiff_threshold:
            return False   # không có gì thay đổi

        # MOG2: xác nhận người thật hay chỉ thay đổi ánh sáng
        mog2_mask  = self._mog2.apply(gray)
        mog2_score = float(mog2_mask.mean()) / 2.55   # normalize 0–100
        return mog2_score >= self._mog2_threshold
