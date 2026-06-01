"""
Unit tests for core/recognizer.py — FaceRecognizer pipeline.

All model dependencies (YOLO, dlib, SFace) are mocked.
Tests cover: recognition voting logic, IOU cache, confusion margin,
HOG fallback warning, headless flag, backend selection, and
compatibility shim (detect_faces_yolo).
"""

import os
import pickle
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.detector import FaceDetection
from core.recognizer import FaceRecognizer, _build_backends


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BLACK_FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _enc(value: float = 0.5) -> np.ndarray:
    """128-d float32 encoding filled with a constant value."""
    return np.full(128, value, dtype=np.float32)


def _det(x1=10, y1=10, x2=80, y2=80, conf=0.9) -> FaceDetection:
    return FaceDetection(bbox=(x1, y1, x2, y2), confidence=conf)


def _make_recognizer(tmp_path, encodings=None, names=None, backend="dlib") -> FaceRecognizer:
    """Build FaceRecognizer with mocked detector + embedder, optional pre-loaded encodings."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"fake")

    enc_file = tmp_path / "encodings.pkl"
    if encodings is not None:
        # Safe: pkl written by test code within tmp_path
        with open(enc_file, "wb") as f:
            pickle.dump({
                "encodings": encodings,
                "names": names or [],
                "backend": backend,
            }, f)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = None
    mock_embedder.batch_distance.return_value = np.array([])

    with patch("core.recognizer._build_backends", return_value=(mock_detector, mock_embedder)):
        recognizer = FaceRecognizer(
            dataset_path=str(dataset_dir),
            model_path=str(model_file),
            encodings_file=str(enc_file) if encodings is not None else str(enc_file),
            headless=True,
            face_backend=backend,
        )

    # Inject mocks for direct access in tests
    recognizer._mock_detector = mock_detector
    recognizer._mock_embedder = mock_embedder
    return recognizer


# ---------------------------------------------------------------------------
# _build_backends factory
# ---------------------------------------------------------------------------

class TestBuildBackends:
    def test_dlib_backend_returns_detector_and_embedder(self, tmp_path):
        """dlib backend must return (FaceDetector, FaceEmbedder) pair."""
        model = str(tmp_path / "model.onnx")
        Path(model).write_bytes(b"fake")

        # YoloDetector/DlibEmbedder are imported lazily inside _build_backends;
        # patch them at their source module, not at core.recognizer.
        with patch("core.detector.YoloDetector") as mock_yolo_cls, \
             patch("core.embedder.DlibEmbedder") as mock_dlib_cls:
            mock_yolo_cls.return_value = MagicMock()
            mock_dlib_cls.return_value = MagicMock()
            from core.recognizer import _build_backends as _bb
            detector, embedder = _bb("dlib", model, 0.3, 416)
            assert detector is not None
            assert embedder is not None

    def test_sface_backend_returns_detector_and_embedder(self, tmp_path):
        """sface backend must return (FaceDetector, FaceEmbedder) pair."""
        model = str(tmp_path / "model.onnx")
        Path(model).write_bytes(b"fake")

        with patch("core.detector.YuNetDetector") as mock_yunet, \
             patch("core.embedder.SFaceEmbedder") as mock_sface:
            mock_yunet.return_value = MagicMock()
            mock_sface.return_value = MagicMock()
            from core.recognizer import _build_backends as _bb
            detector, embedder = _bb("sface", model, 0.6, 416)
            assert detector is not None
            assert embedder is not None

    def test_unknown_backend_falls_back_to_dlib(self, tmp_path):
        """Any unrecognised backend name should fall back to dlib (else branch)."""
        model = str(tmp_path / "model.onnx")
        Path(model).write_bytes(b"fake")

        with patch("core.detector.YoloDetector") as mock_yolo, \
             patch("core.embedder.DlibEmbedder") as mock_dlib:
            mock_yolo.return_value = MagicMock()
            mock_dlib.return_value = MagicMock()
            from core.recognizer import _build_backends as _bb
            detector, embedder = _bb("unknown_backend", model, 0.3, 416)
            assert detector is not None
            assert embedder is not None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_initializes_with_mocked_backends(self, tmp_path):
        r = _make_recognizer(tmp_path)
        assert r is not None

    def test_loads_encodings_from_file(self, tmp_path):
        encs = [_enc(0.1), _enc(0.9)]
        names = ["Alice", "Bob"]
        r = _make_recognizer(tmp_path, encodings=encs, names=names)
        assert len(r.known_face_encodings) == 2
        assert r.known_face_names == ["Alice", "Bob"]

    def test_unique_person_count_set_correctly(self, tmp_path):
        encs = [_enc(0.1), _enc(0.2), _enc(0.9)]
        names = ["Alice", "Alice", "Bob"]
        r = _make_recognizer(tmp_path, encodings=encs, names=names)
        assert r._unique_person_count == 2

    def test_backend_mismatch_warns(self, tmp_path, caplog):
        encs = [_enc(0.5)]
        names = ["Alice"]
        # Save pkl with "sface" tag but load with "dlib" config
        enc_file = tmp_path / "encodings.pkl"
        with open(enc_file, "wb") as f:
            pickle.dump({"encodings": encs, "names": names, "backend": "sface"}, f)

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        mock_d = MagicMock()
        mock_d.detect.return_value = []
        mock_e = MagicMock()
        mock_e.encode.return_value = None

        with patch("core.recognizer._build_backends", return_value=(mock_d, mock_e)):
            with caplog.at_level(logging.WARNING, logger="core.recognizer"):
                r = FaceRecognizer(
                    dataset_path=str(dataset_dir),
                    model_path=str(model_file),
                    encodings_file=str(enc_file),
                    face_backend="dlib",
                    headless=True,
                )
        assert any("backend" in msg.lower() for msg in caplog.messages)

    def test_headless_flag_stored(self, tmp_path):
        r = _make_recognizer(tmp_path)
        assert r.headless is True

    def test_active_process_every_minimum_one(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        enc_file = tmp_path / "e.pkl"
        with open(enc_file, "wb") as f:
            pickle.dump({"encodings": [], "names": [], "backend": "dlib"}, f)
        mock_d = MagicMock()
        mock_d.detect.return_value = []
        mock_e = MagicMock()
        with patch("core.recognizer._build_backends", return_value=(mock_d, mock_e)):
            r = FaceRecognizer(
                dataset_path=str(dataset_dir), model_path=str(model_file),
                encodings_file=str(enc_file), active_process_every=0,
            )
        assert r.active_process_every >= 1


# ---------------------------------------------------------------------------
# IOU helpers
# ---------------------------------------------------------------------------

class TestIOU:
    def test_iou_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert FaceRecognizer._iou(box, box) == pytest.approx(1.0)

    def test_iou_non_overlapping(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert FaceRecognizer._iou(box1, box2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        box1 = (0, 0, 10, 10)  # area = 100
        box2 = (5, 5, 15, 15)  # area = 100, overlap = 25
        iou = FaceRecognizer._iou(box1, box2)
        # intersection=25, union=175
        assert iou == pytest.approx(25 / 175, abs=1e-4)

    def test_iou_symmetric(self):
        box1 = (10, 20, 40, 60)
        box2 = (25, 35, 55, 70)
        assert FaceRecognizer._iou(box1, box2) == pytest.approx(
            FaceRecognizer._iou(box2, box1)
        )

    def test_iou_zero_area_box(self):
        box1 = (10, 10, 10, 10)   # zero area
        box2 = (10, 10, 20, 20)
        assert FaceRecognizer._iou(box1, box2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# IOU cache — _get_cached_name / _update_cache
# ---------------------------------------------------------------------------

class TestIouCache:
    def test_cache_miss_returns_none(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        result = r._get_cached_name((10, 10, 50, 50))
        assert result is None

    def test_cache_hit_same_bbox_same_frame(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        r._update_cache((10, 10, 50, 50), "Alice")
        r._frame_count = 1   # within recognition_interval
        result = r._get_cached_name((10, 10, 50, 50))
        assert result == "Alice"

    def test_cache_miss_after_interval(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        r._update_cache((10, 10, 50, 50), "Alice")
        r._frame_count = r.recognition_interval + 1   # expired
        result = r._get_cached_name((10, 10, 50, 50))
        assert result is None

    def test_cache_hit_high_iou_bbox(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        r._update_cache((10, 10, 50, 50), "Alice")
        r._frame_count = 1
        # Slightly shifted bbox, high IOU → should hit
        result = r._get_cached_name((11, 11, 51, 51))
        assert result == "Alice"

    def test_cache_miss_low_iou_bbox(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        r._update_cache((10, 10, 30, 30), "Alice")
        r._frame_count = 1
        # Far-away bbox, low IOU → miss
        result = r._get_cached_name((200, 200, 250, 250))
        assert result is None

    def test_update_cache_replaces_same_bbox(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._frame_count = 0
        r._update_cache((10, 10, 50, 50), "Alice")
        r._update_cache((10, 10, 50, 50), "Bob")
        r._frame_count = 1
        result = r._get_cached_name((10, 10, 50, 50))
        assert result == "Bob"


# ---------------------------------------------------------------------------
# recognize_face_in_region — voting logic
# ---------------------------------------------------------------------------

class TestRecognizeFaceInRegion:
    def _make_r_with_encodings(self, tmp_path, encs, names) -> FaceRecognizer:
        r = _make_recognizer(tmp_path, encodings=encs, names=names)
        r.known_face_encodings = encs
        r.known_face_names = names
        r._unique_person_count = len(set(names))
        return r

    def test_returns_no_face_when_encode_returns_none(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._mock_embedder.encode.return_value = None
        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result == "No Face"

    def test_returns_no_training_data_when_empty_db(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._mock_embedder.encode.return_value = _enc(0.5)
        r.known_face_encodings = []
        r.known_face_names = []
        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result == "No Training Data"

    def test_returns_correct_winner(self, tmp_path):
        encs = [_enc(0.1), _enc(0.9)]
        names = ["Alice", "Bob"]
        r = self._make_r_with_encodings(tmp_path, encs, names)

        # Query closer to Alice (distance 0.0) vs Bob (distance 0.8)
        r._mock_embedder.encode.return_value = _enc(0.1)
        r._mock_embedder.batch_distance.return_value = np.array([0.0, 0.8])
        r.tolerance = 0.6

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result.startswith("Alice")

    def test_returns_unknown_when_all_distances_exceed_tolerance(self, tmp_path):
        encs = [_enc(0.1), _enc(0.9)]
        names = ["Alice", "Bob"]
        r = self._make_r_with_encodings(tmp_path, encs, names)
        r._mock_embedder.encode.return_value = _enc(0.5)
        r._mock_embedder.batch_distance.return_value = np.array([0.9, 0.9])
        r.tolerance = 0.5

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result == "Unknown"

    def test_confusion_margin_returns_unknown_when_too_close(self, tmp_path):
        encs = [_enc(0.1), _enc(0.15)]   # Alice and Bob are very close
        names = ["Alice", "Bob"]
        r = self._make_r_with_encodings(tmp_path, encs, names)
        r._mock_embedder.encode.return_value = _enc(0.12)
        # Alice at 0.10, Bob at 0.08 → margin = 0.02 < confusion_margin
        r._mock_embedder.batch_distance.return_value = np.array([0.10, 0.08])
        r.tolerance = 0.5
        r.confusion_margin = 0.15   # require at least 0.15 gap

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result == "Unknown"

    def test_confusion_margin_allows_clear_winner(self, tmp_path):
        encs = [_enc(0.1), _enc(0.9)]
        names = ["Alice", "Bob"]
        r = self._make_r_with_encodings(tmp_path, encs, names)
        r._mock_embedder.encode.return_value = _enc(0.1)
        r._mock_embedder.batch_distance.return_value = np.array([0.05, 0.75])
        r.tolerance = 0.5
        r.confusion_margin = 0.10   # gap = 0.70 → clear win

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result.startswith("Alice")

    def test_result_includes_confidence_score(self, tmp_path):
        encs = [_enc(0.1)]
        names = ["Alice"]
        r = self._make_r_with_encodings(tmp_path, encs, names)
        r._mock_embedder.encode.return_value = _enc(0.1)
        r._mock_embedder.batch_distance.return_value = np.array([0.1])  # close
        r.tolerance = 0.5
        r._unique_person_count = 1  # only 1 person → no margin check

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert "(" in result and ")" in result   # e.g. "Alice (0.90)"

    def test_weighted_voting_prefers_closer_encoding(self, tmp_path):
        """Weighted vote: single very-close encoding beats two farther ones.

        Count-based voting: Alice wins (2 entries vs 1).
        Weighted voting: Bob wins because 1.0 > (1-0.57)+(1-0.57) = 0.86.
        """
        encs = [_enc(0.5), _enc(0.6), _enc(0.1)]
        names = ["Alice", "Alice", "Bob"]
        r = self._make_r_with_encodings(tmp_path, encs, names)
        r._mock_embedder.encode.return_value = _enc(0.1)
        # Bob dist=0.0 → weight=1.0; Alice dists=0.57,0.57 → weight=0.43+0.43=0.86
        r._mock_embedder.batch_distance.return_value = np.array([0.57, 0.57, 0.0])
        r.tolerance = 0.6
        r.confusion_margin = 0.05
        r.recognition_top_k = 3

        result = r.recognize_face_in_region(BLACK_FRAME, _det())
        assert result.startswith("Bob")


# ---------------------------------------------------------------------------
# HOG fallback warning
# ---------------------------------------------------------------------------

class TestHogFallback:
    def test_fallback_increments_count(self, tmp_path):
        r = _make_recognizer(tmp_path)
        assert r._hog_fallback_count == 0

        enc = np.ones(128, dtype=np.float32)
        with patch("face_recognition.face_encodings", return_value=[enc]):
            r._fallback_extract_encoding(BLACK_FRAME)

        assert r._hog_fallback_count == 1

    def test_fallback_logs_warning(self, tmp_path, caplog):
        r = _make_recognizer(tmp_path)
        enc = np.ones(128, dtype=np.float32)
        with patch("face_recognition.face_encodings", return_value=[enc]):
            with caplog.at_level(logging.WARNING):
                r._fallback_extract_encoding(BLACK_FRAME)
        assert any("HOG" in msg or "fallback" in msg.lower() for msg in caplog.messages)

    def test_fallback_returns_encoding(self, tmp_path):
        r = _make_recognizer(tmp_path)
        enc = np.ones(128, dtype=np.float32)
        with patch("face_recognition.face_encodings", return_value=[enc]):
            result = r._fallback_extract_encoding(BLACK_FRAME)
        assert result is not None
        assert result.shape == (128,)

    def test_fallback_returns_none_when_no_face(self, tmp_path):
        r = _make_recognizer(tmp_path)
        with patch("face_recognition.face_encodings", return_value=[]):
            result = r._fallback_extract_encoding(BLACK_FRAME)
        assert result is None

    def test_fallback_graceful_when_import_missing(self, tmp_path):
        """After Phase 3, face_recognition removed — should return None silently."""
        r = _make_recognizer(tmp_path)
        with patch.dict("sys.modules", {"face_recognition": None}):
            result = r._fallback_extract_encoding(BLACK_FRAME)
        assert result is None

    def test_fallback_count_not_incremented_when_no_face(self, tmp_path):
        r = _make_recognizer(tmp_path)
        with patch("face_recognition.face_encodings", return_value=[]):
            r._fallback_extract_encoding(BLACK_FRAME)
        assert r._hog_fallback_count == 0


# ---------------------------------------------------------------------------
# detect_faces_yolo compatibility shim
# ---------------------------------------------------------------------------

class TestDetectFacesYoloShim:
    def test_returns_list_of_tuples(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._mock_detector.detect.return_value = [
            _det(x1=10, y1=20, x2=80, y2=90, conf=0.85)
        ]
        result = r.detect_faces_yolo(BLACK_FRAME)
        assert isinstance(result, list)
        assert len(result) == 1
        x1, y1, x2, y2, conf = result[0]
        assert (x1, y1, x2, y2) == (10, 20, 80, 90)
        assert conf == pytest.approx(0.85)

    def test_returns_empty_when_no_detections(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._mock_detector.detect.return_value = []
        result = r.detect_faces_yolo(BLACK_FRAME)
        assert result == []

    def test_delegates_to_detector(self, tmp_path):
        r = _make_recognizer(tmp_path)
        r._mock_detector.detect.return_value = []
        r.detect_faces_yolo(BLACK_FRAME)
        r._mock_detector.detect.assert_called_once_with(BLACK_FRAME)
