#!/usr/bin/env python3
"""
AlertManager — xử lý cảnh báo khi phát hiện người lạ.

Khi RecognitionStabilizer xác nhận có người lạ (Unknown stable):
  1. Kiểm tra cooldown (tránh spam alert)
  2. Lưu ảnh chụp vào alerts/YYYYMMDD_HHMMSS_unknown.jpg
  3. In ra console + tiếng beep hệ thống
  4. Gửi ảnh + tin nhắn qua Telegram Bot (nếu đã cấu hình)

Telegram dùng requests.post trực tiếp đến Bot API — không cần thư viện ngoài ngoài requests.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config


@dataclass
class AlertEvent:
    timestamp: float        # unix time
    snapshot_path: str      # đường dẫn tuyệt đối đến ảnh đã lưu
    bbox: tuple             # (x1, y1, x2, y2)


class AlertManager:
    def __init__(
        self,
        snapshot_dir: str = config.ALERT_SNAPSHOT_DIR,
        cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS,
        enable_sound: bool = config.ALERT_ENABLE_SOUND,
        telegram_bot_token: Optional[str] = config.TELEGRAM_BOT_TOKEN,
        telegram_chat_id: Optional[str] = config.TELEGRAM_CHAT_ID,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._snapshot_dir = snapshot_dir
        self._cooldown = cooldown_seconds
        self._enable_sound = enable_sound
        self._telegram_token = telegram_bot_token
        self._telegram_chat_id = telegram_chat_id
        self._logger = logger or logging.getLogger(__name__)
        self._last_alert_time: float = 0.0

        os.makedirs(snapshot_dir, exist_ok=True)

        if telegram_bot_token and telegram_chat_id:
            self._logger.info("AlertManager: Telegram configured (chat_id=%s)", telegram_chat_id)
        else:
            self._logger.info("AlertManager: Telegram not configured — local only")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> Optional[AlertEvent]:
        """
        Kích hoạt alert nếu ngoài cooldown window.
        Luôn lưu snapshot + log console.
        Gửi Telegram nếu token đã cấu hình.
        Returns AlertEvent nếu alert thực sự fired, None nếu bị suppressed.
        """
        if self.is_in_cooldown():
            remaining = self._cooldown - (time.time() - self._last_alert_time)
            self._logger.debug("Alert suppressed (cooldown %.0fs remaining)", remaining)
            return None

        self._last_alert_time = time.time()

        snapshot_path = self._save_snapshot(frame, bbox)

        self._logger.warning(
            "INTRUDER ALERT — Unknown person detected! Snapshot: %s", snapshot_path
        )

        if self._enable_sound:
            self._play_sound()

        if self._telegram_token and self._telegram_chat_id:
            self._send_telegram(snapshot_path)

        return AlertEvent(
            timestamp=self._last_alert_time,
            snapshot_path=snapshot_path,
            bbox=bbox,
        )

    def is_in_cooldown(self) -> bool:
        return (time.time() - self._last_alert_time) < self._cooldown

    @property
    def last_alert_time(self) -> float:
        return self._last_alert_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_snapshot(self, frame: np.ndarray, bbox: tuple) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_unknown.jpg"
        path = os.path.join(self._snapshot_dir, filename)

        # Draw a minimal red rectangle so the face is obvious in the saved image
        x1, y1, x2, y2 = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)
        cv2.putText(annotated, "UNKNOWN", (x1, max(y1 - 10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

        cv2.imwrite(path, annotated)
        return path

    def _play_sound(self) -> None:
        # \a (BEL) works in most terminals. Falls back silently if not supported.
        print("\a", end="", flush=True)

    def _send_telegram(self, snapshot_path: str) -> None:
        try:
            import requests  # optional dependency
        except ImportError:
            self._logger.debug("requests not installed — Telegram alert skipped")
            return

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendPhoto"
        caption = (
            f"⚠️ CẢNH BÁO: Phát hiện người lạ!\n"
            f"🕐 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
        )
        try:
            with open(snapshot_path, "rb") as photo:
                resp = requests.post(
                    url,
                    data={"chat_id": self._telegram_chat_id, "caption": caption},
                    files={"photo": photo},
                    timeout=10,
                )
            if resp.status_code == 200:
                self._logger.info("Telegram alert sent")
            else:
                self._logger.warning(
                    "Telegram API error %d: %s", resp.status_code, resp.text[:200]
                )
        except Exception as e:
            self._logger.warning("Telegram send failed: %s", e)
