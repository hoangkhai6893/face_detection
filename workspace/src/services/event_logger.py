#!/usr/bin/env python3
"""
EventLogger — ghi lịch sử nhận diện vào file JSON Lines với daily rotation.

Mỗi sự kiện nhận diện được ghi thành 1 dòng JSON vào file log theo ngày:
  logs/entry_log_YYYY-MM-DD.jsonl

Thread-safe: dùng threading.Lock để tránh xung đột khi ghi.
Rotation: tạo file mới mỗi ngày, xóa file cũ hơn keep_days ngày.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

import config


@dataclass
class RecognitionEvent:
    timestamp_iso: str           # "2026-05-30T14:32:01.123"
    person_name:   str           # "Khai", "Unknown", ...
    confidence:    Optional[float]  # 0.0–1.0 if known, None if Unknown
    bbox:          Optional[List[int]]  # [x1, y1, x2, y2]


class EventLogger:
    def __init__(
        self,
        log_path: str = config.ENTRY_LOG_PATH,
        logger: Optional[logging.Logger] = None,
        keep_days: int = 30,
    ) -> None:
        self._log_dir   = os.path.dirname(log_path) or "."
        # Stem: "entry_log" from "entry_log.jsonl", or any custom name
        self._log_stem  = os.path.splitext(os.path.basename(log_path))[0]
        self._lock      = threading.Lock()
        self._logger    = logger or logging.getLogger(__name__)
        self._keep_days = keep_days

        os.makedirs(self._log_dir, exist_ok=True)
        self._log_path = self._daily_path()  # current day's log file
        self._cleanup_old_logs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        person_name: str,
        bbox: tuple,
        confidence: Optional[float] = None,
    ) -> RecognitionEvent:
        """Log one recognition event. Thread-safe."""
        event = RecognitionEvent(
            timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
            person_name=person_name,
            confidence=round(confidence, 4) if confidence is not None else None,
            bbox=list(bbox) if bbox else None,
        )
        self._append_line(event)
        self._logger.debug(
            "Entry logged: %s (conf=%.2f)" if confidence else "Entry logged: %s",
            person_name,
            *(([confidence]) if confidence else []),
        )
        return event

    def read_recent(self, n: int = 50) -> List[dict]:
        """Read the n most recent entries from today's log file."""
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-n:] if line.strip()]
        except FileNotFoundError:
            return []
        except Exception as e:
            self._logger.warning("Error reading log: %s", e)
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _daily_path(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self._log_dir, f"{self._log_stem}_{today}.jsonl")

    def _append_line(self, event: RecognitionEvent) -> None:
        row  = asdict(event)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            # Rotate to new daily file if date changed since last write
            current_path = self._daily_path()
            if current_path != self._log_path:
                self._log_path = current_path
                self._cleanup_old_logs()
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def _cleanup_old_logs(self) -> None:
        """Remove log files older than keep_days days."""
        cutoff = time.time() - self._keep_days * 86400
        try:
            for fname in os.listdir(self._log_dir):
                if not (fname.startswith(self._log_stem + "_") and fname.endswith(".jsonl")):
                    continue
                fpath = os.path.join(self._log_dir, fname)
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    self._logger.info("Removed old log: %s", fname)
        except OSError:
            pass
