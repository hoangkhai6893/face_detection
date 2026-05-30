"""
Unit tests for AlertManager.
"""
import os
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from alert_manager import AlertManager


BBOX = (10, 20, 100, 120)


@pytest.fixture
def manager(tmp_path):
    return AlertManager(
        snapshot_dir=str(tmp_path / "alerts"),
        cooldown_seconds=5.0,
        enable_sound=False,
        telegram_bot_token=None,
        telegram_chat_id=None,
    )


@pytest.fixture
def frame():
    return np.zeros((200, 200, 3), dtype=np.uint8)


class TestTrigger:
    def test_trigger_saves_snapshot(self, manager, frame, tmp_path):
        event = manager.trigger(frame, BBOX)
        assert event is not None
        assert os.path.exists(event.snapshot_path)

    def test_snapshot_filename_pattern(self, manager, frame):
        event = manager.trigger(frame, BBOX)
        name = os.path.basename(event.snapshot_path)
        # Pattern: YYYYMMDD_HHMMSS_unknown.jpg
        import re
        assert re.match(r"\d{8}_\d{6}_unknown\.jpg", name)

    def test_cooldown_suppresses_second_trigger(self, manager, frame):
        first = manager.trigger(frame, BBOX)
        second = manager.trigger(frame, BBOX)
        assert first is not None
        assert second is None  # suppressed

    def test_cooldown_allows_after_expiry(self, manager, frame):
        manager.trigger(frame, BBOX)
        # Manually set last alert time to the past
        manager._last_alert_time -= 10.0
        result = manager.trigger(frame, BBOX)
        assert result is not None

    def test_snapshot_dir_created_if_missing(self, tmp_path, frame):
        new_dir = str(tmp_path / "new" / "alerts")
        m = AlertManager(snapshot_dir=new_dir, cooldown_seconds=0, enable_sound=False)
        event = m.trigger(frame, BBOX)
        assert os.path.isdir(new_dir)
        assert event is not None


class TestTelegram:
    def test_telegram_skipped_when_no_token(self, manager, frame):
        with patch("alert_manager.AlertManager._send_telegram") as mock_send:
            manager.trigger(frame, BBOX)
            mock_send.assert_not_called()

    def test_telegram_called_when_configured(self, tmp_path, frame):
        m = AlertManager(
            snapshot_dir=str(tmp_path / "alerts"),
            cooldown_seconds=0,
            enable_sound=False,
            telegram_bot_token="fake_token",
            telegram_chat_id="123456",
        )
        with patch("alert_manager.AlertManager._send_telegram") as mock_send:
            m.trigger(frame, BBOX)
            mock_send.assert_called_once()


class TestCooldown:
    def test_is_in_cooldown_true_immediately(self, manager, frame):
        manager.trigger(frame, BBOX)
        assert manager.is_in_cooldown() is True

    def test_is_in_cooldown_false_initially(self, manager):
        assert manager.is_in_cooldown() is False
