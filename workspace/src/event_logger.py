#!/usr/bin/env python3
"""
EventLogger — ghi lịch sử nhận diện vào file JSON Lines.

Mỗi sự kiện nhận diện được ghi thành 1 dòng JSON vào ENTRY_LOG_PATH.
Thread-safe: dùng threading.Lock để tránh xung đột khi ghi.

Format JSON Lines (.jsonl): mỗi dòng là 1 JSON object độc lập.
Ưu điểm: append O(1), không cần đọc lại toàn file.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

import config


@dataclass
class RecognitionEvent:
    timestamp_iso: str          # "2026-05-30T14:32:01.123"
    person_name: str            # "Khai", "Unknown", ...
    confidence: Optional[float] # 0.0–1.0 nếu known, None nếu Unknown
    bbox: Optional[List[int]]   # [x1, y1, x2, y2]


class EventLogger:
    def __init__(
        self,
        log_path: str = config.ENTRY_LOG_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._log_path = log_path
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger(__name__)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(
        self,
        person_name: str,
        bbox: tuple,
        confidence: Optional[float] = None,
    ) -> RecognitionEvent:
        """
        Ghi 1 sự kiện nhận diện vào log file.
        Thread-safe, tạo file + thư mục nếu chưa tồn tại.
        """
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
        """Đọc n dòng cuối từ log file. Trả về list of dicts."""
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [json.loads(line) for line in lines[-n:] if line.strip()]
        except FileNotFoundError:
            return []
        except Exception as e:
            self._logger.warning("Error reading log: %s", e)
            return []

    def _append_line(self, event: RecognitionEvent) -> None:
        row = asdict(event)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)
