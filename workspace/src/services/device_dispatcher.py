#!/usr/bin/env python3
"""
DeviceDispatcher — kích hoạt thiết bị nhà thông minh khi nhận diện thành viên.

Kiến trúc plugin:
  Backend (ABC)
    ├── LogBackend      — in ra console, luôn available, dùng khi test
    ├── MqttBackend     — publish lên MQTT broker (paho-mqtt)
    ├── WebhookBackend  — HTTP POST đến URL tùy chỉnh (requests)
    └── SerialBackend   — gửi lệnh đến Arduino/ESP32 qua cổng serial (pyserial)

Mỗi thành viên có file profile.json trong family_images/<Person>/profile.json
chứa danh sách thiết bị cần kích hoạt và backend tương ứng.

Thiếu thư viện paho-mqtt hoặc pyserial → backend đó is_available()=False,
toàn bộ app không crash — chỉ log warning và bỏ qua.
"""

import abc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceAction:
    id: str                              # logical device id, e.g. "light_living"
    action: str                          # semantic action, e.g. "on", "off", "unlock"
    backend: str                         # "mqtt", "webhook", "serial", "log"
    params: Dict[str, Any] = field(default_factory=dict)
    # backend-specific keys in params:
    #   mqtt:    topic, payload_template (optional)
    #   webhook: url, extra (optional dict merged into POST body)
    #   serial:  command (optional, default: "<id>:<action>\n")
    #   log:     (none)


@dataclass
class PersonProfile:
    name: str
    welcome_message: str
    devices: List[DeviceAction]
    recognition_cooldown_override: Optional[float] = None  # overrides DISPATCH_COOLDOWN_SECONDS

    @classmethod
    def load(cls, profile_path: str) -> "PersonProfile":
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        devices = [
            DeviceAction(
                id=d["id"],
                action=d["action"],
                backend=d.get("backend", "log"),
                params={k: v for k, v in d.items() if k not in ("id", "action", "backend")},
            )
            for d in data.get("devices", [])
        ]
        return cls(
            name=data.get("name", ""),
            welcome_message=data.get("welcome_message", ""),
            devices=devices,
            recognition_cooldown_override=data.get("recognition_cooldown_override"),
        )

    @classmethod
    def load_for_person(
        cls,
        person_name: str,
        dataset_path: str = config.DATASET_PATH,
    ) -> Optional["PersonProfile"]:
        path = os.path.join(dataset_path, person_name, "profile.json")
        if not os.path.exists(path):
            return None
        try:
            return cls.load(path)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load profile for %s: %s", person_name, e
            )
            return None


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class Backend(abc.ABC):
    """Interface for all device control backends."""

    @abc.abstractmethod
    def trigger(self, device_id: str, action: str, params: Dict[str, Any]) -> bool:
        """Fire the action. Returns True on success, False on error."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is usable."""


class LogBackend(Backend):
    """No-op backend — print to console. Always available. Good for testing."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def trigger(self, device_id: str, action: str, params: Dict[str, Any]) -> bool:
        self._logger.info("[LogBackend] device=%s action=%s params=%s", device_id, action, params)
        return True

    def is_available(self) -> bool:
        return True


class MqttBackend(Backend):
    """Publish device action to MQTT broker. Lazy-connects on first use."""

    def __init__(
        self,
        host: str = config.MQTT_HOST,
        port: int = config.MQTT_PORT,
        topic_prefix: str = config.MQTT_TOPIC_PREFIX,
        username: Optional[str] = config.MQTT_USERNAME,
        password: Optional[str] = config.MQTT_PASSWORD,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._topic_prefix = topic_prefix
        self._username = username
        self._password = password
        self._logger = logger or logging.getLogger(__name__)
        self._client = None

    def trigger(self, device_id: str, action: str, params: Dict[str, Any]) -> bool:
        if not self._ensure_connected():
            return False
        topic = params.get("topic") or f"{self._topic_prefix}/{device_id}"
        template = params.get("payload_template")
        if template:
            payload = template.replace("{action}", action).replace("{device_id}", device_id)
        else:
            payload = json.dumps({"device": device_id, "action": action})
        try:
            result = self._client.publish(topic, payload, qos=1)
            ok = result.rc == 0
            if ok:
                self._logger.info("[MQTT] %s → %s", topic, payload)
            else:
                self._logger.warning("[MQTT] publish failed rc=%d topic=%s", result.rc, topic)
            return ok
        except Exception as e:
            self._logger.warning("[MQTT] publish error: %s", e)
            return False

    def is_available(self) -> bool:
        try:
            import paho.mqtt.client  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_connected(self) -> bool:
        if self._client is not None:
            return True
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self._logger.warning("[MQTT] paho-mqtt not installed. Run: pip install paho-mqtt")
            return False
        try:
            client = mqtt.Client()
            if self._username:
                client.username_pw_set(self._username, self._password)
            client.connect(self._host, self._port, keepalive=60)
            client.loop_start()
            self._client = client
            self._logger.info("[MQTT] Connected to %s:%d", self._host, self._port)
            return True
        except Exception as e:
            self._logger.warning("[MQTT] Connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None


class WebhookBackend(Backend):
    """HTTP POST to a URL. Uses requests library."""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._logger = logger or logging.getLogger(__name__)

    def trigger(self, device_id: str, action: str, params: Dict[str, Any]) -> bool:
        url = params.get("url")
        if not url:
            self._logger.warning("[Webhook] No URL in params for device=%s", device_id)
            return False
        try:
            import requests
        except ImportError:
            self._logger.warning("[Webhook] requests not installed. Run: pip install requests")
            return False
        body = {"device": device_id, "action": action}
        body.update(params.get("extra") or {})
        try:
            resp = requests.post(url, json=body, timeout=self._timeout)
            ok = resp.status_code < 400
            self._logger.info("[Webhook] POST %s → %d", url, resp.status_code)
            return ok
        except Exception as e:
            self._logger.warning("[Webhook] POST failed: %s", e)
            return False

    def is_available(self) -> bool:
        return True  # errors are soft — always attempt


class SerialBackend(Backend):
    """Send command to Arduino/ESP32 via serial port. Lazy-opens on first use."""

    def __init__(
        self,
        port: str = config.SERIAL_PORT,
        baud_rate: int = config.SERIAL_BAUD_RATE,
        timeout_seconds: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._port = port
        self._baud = baud_rate
        self._timeout = timeout_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._serial = None

    def trigger(self, device_id: str, action: str, params: Dict[str, Any]) -> bool:
        if not self._ensure_open():
            return False
        command = params.get("command") or f"{device_id}:{action}\n"
        try:
            self._serial.write(command.encode())
            self._serial.flush()
            self._logger.info("[Serial] %s → %r", self._port, command)
            return True
        except Exception as e:
            self._logger.warning("[Serial] write error: %s", e)
            self._serial = None
            return False

    def is_available(self) -> bool:
        return os.path.exists(self._port)

    def _ensure_open(self) -> bool:
        if self._serial is not None:
            return True
        if not os.path.exists(self._port):
            self._logger.debug("[Serial] port not found: %s", self._port)
            return False
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baud, timeout=self._timeout)
            self._logger.info("[Serial] Opened %s at %d baud", self._port, self._baud)
            return True
        except ImportError:
            self._logger.warning("[Serial] pyserial not installed. Run: pip install pyserial")
            return False
        except Exception as e:
            self._logger.warning("[Serial] open error: %s", e)
            return False

    def close(self) -> None:
        if self._serial:
            self._serial.close()
            self._serial = None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class DeviceDispatcher:
    """
    Reads a person's profile.json, resolves backends, dispatches all device actions.
    Applies per-person cooldown to avoid re-triggering on every recognition event.
    """

    def __init__(
        self,
        dataset_path: str = config.DATASET_PATH,
        default_cooldown: float = config.DISPATCH_COOLDOWN_SECONDS,
        backends: Optional[Dict[str, Backend]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._dataset_path = dataset_path
        self._default_cooldown = default_cooldown
        self._logger = logger or logging.getLogger(__name__)
        self._backends: Dict[str, Backend] = backends or {
            "log": LogBackend(self._logger),
            "mqtt": MqttBackend(logger=self._logger),
            "webhook": WebhookBackend(logger=self._logger),
            "serial": SerialBackend(logger=self._logger),
        }
        self._last_dispatch: Dict[str, float] = {}  # person_name → unix time

    def dispatch(
        self,
        person_name: str,
        confidence: Optional[float] = None,
    ) -> bool:
        """
        Load profile for person, fire all device actions.
        Returns True if dispatch ran, False if suppressed by cooldown or no profile.
        """
        if self.is_in_cooldown(person_name):
            self._logger.debug("Dispatch suppressed for %s (cooldown)", person_name)
            return False

        profile = PersonProfile.load_for_person(person_name, self._dataset_path)
        if profile is None:
            self._logger.debug("No profile.json for %s — skipping dispatch", person_name)
            return False

        self._last_dispatch[person_name] = time.time()

        if profile.welcome_message:
            self._logger.info("[Dispatch] %s", profile.welcome_message)

        success = True
        for device in profile.devices:
            ok = self._fire_action(device)
            if not ok:
                success = False

        self._logger.info(
            "[Dispatch] %s — %d device(s) fired (conf=%.2f)",
            person_name, len(profile.devices), confidence or 0.0,
        )
        return success

    def is_in_cooldown(self, person_name: str) -> bool:
        last = self._last_dispatch.get(person_name, 0.0)
        return (time.time() - last) < self._default_cooldown

    def shutdown(self) -> None:
        """Clean up backend connections (call on app exit)."""
        if isinstance(self._backends.get("mqtt"), MqttBackend):
            self._backends["mqtt"].disconnect()
        if isinstance(self._backends.get("serial"), SerialBackend):
            self._backends["serial"].close()

    def _fire_action(self, action: DeviceAction) -> bool:
        backend = self._backends.get(action.backend)
        if backend is None:
            self._logger.warning("Unknown backend '%s' for device %s", action.backend, action.id)
            return False
        if not backend.is_available():
            self._logger.debug("Backend '%s' not available — skipping %s", action.backend, action.id)
            return False
        return backend.trigger(action.id, action.action, action.params)
