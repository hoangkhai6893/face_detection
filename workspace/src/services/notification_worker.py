#!/usr/bin/env python3
"""
NotificationWorker — background thread cho tất cả I/O side-effects.

Camera loop gọi .submit(event) không bao giờ block: < 1ms.
Background thread nhận StableEvent từ queue rồi gọi:
  EventLogger.log()         — disk write (~1ms)
  DeviceDispatcher.dispatch() — MQTT / Serial / Webhook (~10-500ms)
  AlertManager.trigger()    — Telegram POST (~500ms-5s)

Queue có giới hạn (queue_size=20). Nếu đầy → drop event cũ nhất (không block loop).
"""

import logging
import queue
import sys
import os
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.recognition_stabilizer import StableEvent


class NotificationWorker:
    """
    Tạo 1 daemon thread xử lý I/O bất đồng bộ.

    Usage:
        worker = NotificationWorker(alert_manager, event_logger, dispatcher)
        # trong callback camera loop:
        worker.submit(stable_event)
        # khi tắt app:
        worker.shutdown()
    """

    def __init__(
        self,
        alert_manager=None,
        event_logger=None,
        dispatcher=None,
        queue_size: int = 20,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._alert_manager = alert_manager
        self._event_logger = event_logger
        self._dispatcher = dispatcher
        self._logger = logger or logging.getLogger(__name__)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="NotificationWorker"
        )
        self._thread.start()
        self._logger.info("NotificationWorker: started (queue_size=%d)", queue_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, event: StableEvent) -> None:
        """
        Đẩy StableEvent vào queue. Non-blocking.
        Nếu queue đầy → drop event cũ nhất để nhường chỗ cho event mới.
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._logger.debug("NotificationWorker: event dropped (queue still full)")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Chờ xử lý xong event cuối rồi dừng thread. Gọi khi tắt app."""
        self._queue.put(None)  # sentinel
        self._thread.join(timeout)
        self._logger.info("NotificationWorker: stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:   # sentinel → shutdown
                break
            try:
                self._dispatch(item)
            except Exception as exc:
                self._logger.error("NotificationWorker dispatch error: %s", exc)

    def _dispatch(self, event: StableEvent) -> None:
        if event.is_known:
            self._logger.info(
                "STABLE KNOWN: %s (conf=%.2f)", event.person_name, event.confidence or 0
            )
            if self._event_logger:
                self._event_logger.log(event.person_name, event.bbox, event.confidence)
            if self._dispatcher:
                self._dispatcher.dispatch(event.person_name, event.confidence)
        else:
            self._logger.warning("STABLE UNKNOWN: bbox=%s", event.bbox)
            if self._alert_manager:
                self._alert_manager.trigger(event.frame, event.bbox)
            if self._event_logger:
                self._event_logger.log("Unknown", event.bbox, confidence=None)
