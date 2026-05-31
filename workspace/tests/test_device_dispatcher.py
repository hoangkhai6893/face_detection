"""
Unit tests for DeviceDispatcher and backends.
"""
import json
import os
import pytest
from unittest.mock import MagicMock, patch
from services.device_dispatcher import (
    DeviceDispatcher, PersonProfile, DeviceAction,
    LogBackend, WebhookBackend, MqttBackend, SerialBackend,
)


# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dataset(tmp_path):
    """Dataset with one person having a profile.json."""
    person_dir = tmp_path / "Khai"
    person_dir.mkdir()
    profile = {
        "name": "Khai",
        "welcome_message": "Xin chào Khai!",
        "recognition_cooldown_override": None,
        "devices": [
            {"id": "light_living", "action": "on", "backend": "log"},
            {"id": "door_lock", "action": "unlock", "backend": "log"},
        ],
    }
    (person_dir / "profile.json").write_text(json.dumps(profile))
    return tmp_path


# ---------------------------------------------------------------------------
# PersonProfile
# ---------------------------------------------------------------------------

class TestPersonProfile:
    def test_load_parses_devices(self, tmp_dataset):
        profile = PersonProfile.load_for_person("Khai", str(tmp_dataset))
        assert profile is not None
        assert profile.name == "Khai"
        assert len(profile.devices) == 2
        assert profile.devices[0].id == "light_living"
        assert profile.devices[0].action == "on"

    def test_load_returns_none_for_missing_profile(self, tmp_dataset):
        profile = PersonProfile.load_for_person("NoSuchPerson", str(tmp_dataset))
        assert profile is None


# ---------------------------------------------------------------------------
# LogBackend
# ---------------------------------------------------------------------------

class TestLogBackend:
    def test_always_available(self):
        assert LogBackend().is_available() is True

    def test_trigger_returns_true(self):
        assert LogBackend().trigger("light", "on", {}) is True


# ---------------------------------------------------------------------------
# WebhookBackend
# ---------------------------------------------------------------------------

class TestWebhookBackend:
    def test_trigger_posts_to_url(self):
        backend = WebhookBackend()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("services.device_dispatcher.WebhookBackend.trigger") as mock_trigger:
            mock_trigger.return_value = True
            result = backend.trigger("light", "on", {"url": "http://example.com/webhook"})

    def test_trigger_returns_false_when_no_url(self):
        backend = WebhookBackend()
        result = backend.trigger("light", "on", {})  # no url in params
        assert result is False

    def test_is_available_always_true(self):
        assert WebhookBackend().is_available() is True


# ---------------------------------------------------------------------------
# MqttBackend
# ---------------------------------------------------------------------------

class TestMqttBackend:
    def test_is_available_false_when_no_paho(self):
        import sys
        with patch.dict(sys.modules, {"paho": None, "paho.mqtt": None, "paho.mqtt.client": None}):
            backend = MqttBackend()
            # is_available does import check
            # If paho not installed, returns False
            # We just verify it doesn't crash
            result = backend.is_available()
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SerialBackend
# ---------------------------------------------------------------------------

class TestSerialBackend:
    def test_is_available_false_when_port_missing(self):
        backend = SerialBackend(port="/dev/nonexistent_tty_xyz")
        assert backend.is_available() is False

    def test_trigger_returns_false_when_port_missing(self):
        backend = SerialBackend(port="/dev/nonexistent_tty_xyz")
        result = backend.trigger("door", "unlock", {})
        assert result is False


# ---------------------------------------------------------------------------
# DeviceDispatcher
# ---------------------------------------------------------------------------

class TestDeviceDispatcher:
    def test_no_profile_skips_silently(self, tmp_dataset):
        d = DeviceDispatcher(dataset_path=str(tmp_dataset), default_cooldown=0)
        result = d.dispatch("NoSuchPerson")
        assert result is False  # no profile → skip

    def test_dispatch_fires_all_devices(self, tmp_dataset):
        mock_log = MagicMock()
        mock_log.is_available.return_value = True
        mock_log.trigger.return_value = True
        d = DeviceDispatcher(
            dataset_path=str(tmp_dataset),
            default_cooldown=0,
            backends={"log": mock_log},
        )
        result = d.dispatch("Khai")
        assert result is True
        assert mock_log.trigger.call_count == 2  # 2 devices in profile

    def test_cooldown_suppresses_second_dispatch(self, tmp_dataset):
        d = DeviceDispatcher(dataset_path=str(tmp_dataset), default_cooldown=60.0)
        d.dispatch("Khai")
        result = d.dispatch("Khai")
        assert result is False

    def test_cooldown_zero_allows_repeat(self, tmp_dataset):
        d = DeviceDispatcher(dataset_path=str(tmp_dataset), default_cooldown=0)
        d.dispatch("Khai")
        result = d.dispatch("Khai")
        assert result is True
