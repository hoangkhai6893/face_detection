"""
Unit tests for EventLogger — logging, thread safety, daily rotation, log cleanup.
"""
import json
import os
import time
import threading
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import patch
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


class TestDailyRotation:
    def test_log_path_contains_today_date(self, tmp_path):
        log = EventLogger(log_path=str(tmp_path / "logs" / "entry_log.jsonl"))
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in log._log_path

    def test_log_path_stem_preserved(self, tmp_path):
        log = EventLogger(log_path=str(tmp_path / "logs" / "my_log.jsonl"))
        assert "my_log_" in log._log_path

    def test_log_file_created_in_log_dir(self, tmp_path):
        log = EventLogger(log_path=str(tmp_path / "logs" / "entry_log.jsonl"))
        log.log("Khai", (10, 20, 80, 90), confidence=0.9)
        # Log file should be inside logs/ directory
        assert os.path.dirname(log._log_path) == str(tmp_path / "logs")

    def test_rotates_to_new_file_when_date_changes(self, tmp_path):
        log = EventLogger(log_path=str(tmp_path / "logs" / "entry_log.jsonl"))
        log.log("Khai", (10, 10, 80, 80), 0.9)
        old_path = log._log_path

        # Simulate date change by patching _daily_path to return a new date
        future_date = "2099-12-31"
        new_path = os.path.join(str(tmp_path / "logs"), f"entry_log_{future_date}.jsonl")
        with patch.object(log, "_daily_path", return_value=new_path):
            log.log("MaiAnh", (10, 10, 80, 80), 0.8)

        # New log file should have been created
        assert os.path.exists(new_path)
        # Old file unchanged
        with open(old_path) as f:
            assert len(f.readlines()) == 1   # only the first log

    def test_different_log_path_per_day(self, tmp_path):
        """Logs on different days go to different files."""
        log_dir = tmp_path / "logs"
        base = str(log_dir / "entry_log.jsonl")

        log = EventLogger(log_path=base)
        path_day1 = log._daily_path()

        # Simulate a different day
        with patch("services.event_logger.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2099, 6, 15)
            path_day2 = log._daily_path()

        assert path_day1 != path_day2
        assert "2099-06-15" in path_day2


class TestLogCleanup:
    def _make_old_log(self, log_dir: str, stem: str, days_old: int) -> str:
        """Create a fake old log file with mtime set to days_old ago."""
        from datetime import date, timedelta
        old_date = (date.today() - timedelta(days=days_old)).strftime("%Y-%m-%d")
        path = os.path.join(log_dir, f"{stem}_{old_date}.jsonl")
        with open(path, "w") as f:
            f.write('{"test": true}\n')
        old_mtime = time.time() - days_old * 86400
        os.utime(path, (old_mtime, old_mtime))
        return path

    def test_old_logs_removed_on_init(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir)
        # Create a log file 60 days old (keep_days=30)
        old_path = self._make_old_log(log_dir, "entry_log", days_old=60)
        assert os.path.exists(old_path)

        # Init EventLogger with keep_days=30 → should remove it
        EventLogger(
            log_path=os.path.join(log_dir, "entry_log.jsonl"),
            keep_days=30,
        )
        assert not os.path.exists(old_path)

    def test_recent_logs_not_removed(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir)
        # Create a log file 10 days old (keep_days=30 → should survive)
        recent_path = self._make_old_log(log_dir, "entry_log", days_old=10)

        EventLogger(
            log_path=os.path.join(log_dir, "entry_log.jsonl"),
            keep_days=30,
        )
        assert os.path.exists(recent_path)

    def test_only_matching_stem_files_removed(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir)
        # An old file with a DIFFERENT stem should NOT be removed
        other_path = os.path.join(log_dir, "other_log_2020-01-01.jsonl")
        with open(other_path, "w") as f:
            f.write("data\n")
        os.utime(other_path, (time.time() - 999 * 86400,) * 2)

        EventLogger(
            log_path=os.path.join(log_dir, "entry_log.jsonl"),
            keep_days=30,
        )
        # other_log_* should NOT be touched (different stem)
        assert os.path.exists(other_path)

    def test_cleanup_tolerates_missing_dir(self, tmp_path):
        """Cleanup should not raise if log dir is empty or nonexistent."""
        log = EventLogger.__new__(EventLogger)
        log._log_dir = str(tmp_path / "nonexistent_dir")
        log._log_stem = "entry_log"
        log._keep_days = 30
        log._logger = __import__("logging").getLogger("test")
        log._cleanup_old_logs()   # should not raise
