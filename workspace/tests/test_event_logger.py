"""
Unit tests for EventLogger.
"""
import json
import threading
import numpy as np
import pytest
from services.event_logger import EventLogger


@pytest.fixture
def logger_instance(tmp_path):
    log_path = str(tmp_path / "logs" / "test_entry.jsonl")
    return EventLogger(log_path=log_path)


BBOX = (10, 20, 100, 120)


class TestLogging:
    def test_creates_file_on_first_log(self, logger_instance, tmp_path):
        logger_instance.log("Khai", BBOX, confidence=0.91)
        log_path = logger_instance._log_path
        assert open(log_path).read().strip() != ""

    def test_appends_multiple_lines(self, logger_instance):
        logger_instance.log("Khai", BBOX, 0.91)
        logger_instance.log("Lan", BBOX, 0.85)
        logger_instance.log("Unknown", BBOX, None)
        lines = open(logger_instance._log_path).readlines()
        assert len(lines) == 3

    def test_json_parseable(self, logger_instance):
        logger_instance.log("Khai", BBOX, 0.91)
        line = open(logger_instance._log_path).readline()
        data = json.loads(line)
        assert data["person_name"] == "Khai"
        assert data["confidence"] == pytest.approx(0.91, abs=0.001)
        assert data["bbox"] == list(BBOX)
        assert "timestamp_iso" in data

    def test_unknown_has_null_confidence(self, logger_instance):
        logger_instance.log("Unknown", BBOX, None)
        data = json.loads(open(logger_instance._log_path).readline())
        assert data["confidence"] is None

    def test_thread_safety(self, logger_instance):
        errors = []

        def worker():
            try:
                for _ in range(10):
                    logger_instance.log("Khai", BBOX, 0.90)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        lines = open(logger_instance._log_path).readlines()
        assert len(lines) == 50  # 5 threads × 10 logs each


class TestReadRecent:
    def test_read_recent_returns_correct_count(self, logger_instance):
        for i in range(20):
            logger_instance.log(f"Person{i}", BBOX, 0.8)
        recent = logger_instance.read_recent(5)
        assert len(recent) == 5
        assert recent[-1]["person_name"] == "Person19"

    def test_read_recent_on_empty_file_returns_empty(self, tmp_path):
        log = EventLogger(log_path=str(tmp_path / "empty.jsonl"))
        assert log.read_recent() == []
