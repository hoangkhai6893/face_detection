"""
Unit tests for encoder.py — _cluster_to_max(), rebuild_encodings(), update_person_encodings().

All YOLO/SFace model deps are mocked so tests run without any model files.
"""

import os
import pickle
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pickle is safe here: all .pkl files are created by our own test code
# within pytest's tmp_path — never loaded from network or untrusted sources.
from core.encoder import (
    _cluster_to_max,
    _extract_face_encoding,
    rebuild_encodings,
    update_person_encodings,
)
import config


# ---------------------------------------------------------------------------
# _cluster_to_max
# ---------------------------------------------------------------------------

class TestClusterToMax:
    def _enc(self, *values) -> np.ndarray:
        """Create a 128-d zero vector with first positions set to given values."""
        v = np.zeros(128, dtype=np.float32)
        for i, val in enumerate(values):
            v[i] = val
        return v

    def test_empty_input_returns_empty(self):
        result = _cluster_to_max([], max_count=10)
        assert result == []

    def test_fewer_than_max_returns_all(self):
        encs = [self._enc(i) for i in range(5)]
        result = _cluster_to_max(encs, max_count=10)
        assert len(result) == 5

    def test_exact_max_returns_all(self):
        encs = [self._enc(i) for i in range(10)]
        result = _cluster_to_max(encs, max_count=10)
        assert len(result) == 10

    def test_result_never_exceeds_max(self):
        encs = [np.random.rand(128).astype(np.float32) for _ in range(100)]
        result = _cluster_to_max(encs, max_count=20)
        assert len(result) <= 20

    def test_near_identical_encodings_capped_at_max(self):
        """100 near-identical encodings (dist << CLUSTERING_THRESHOLD) with max=10 → 10."""
        # Use 100 inputs so len > max_count, ensuring the greedy phase actually runs.
        base = self._enc(0.0)
        near_dups = [base + np.random.randn(128).astype(np.float32) * 0.001
                     for _ in range(100)]
        result = _cluster_to_max(near_dups, max_count=10)
        # Greedy selects ~1 unique; padding fills to exactly max_count.
        assert len(result) == 10

    def test_diverse_encodings_keep_variety(self):
        """Very different encodings should all be kept up to max_count."""
        # 10 encodings very far apart (distance >> CLUSTERING_THRESHOLD)
        encs = [self._enc(i * 10.0) for i in range(10)]
        result = _cluster_to_max(encs, max_count=10)
        assert len(result) == 10

    def test_returns_numpy_arrays(self):
        encs = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        result = _cluster_to_max(encs, max_count=10)
        assert all(isinstance(e, np.ndarray) for e in result)

    def test_max_count_one(self):
        encs = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        result = _cluster_to_max(encs, max_count=1)
        assert len(result) == 1

    def test_fill_uses_evenly_spaced_when_diversity_exhausted(self):
        """When greedy selection gives fewer than max, padding fills to max_count."""
        # 50 near-identical encodings → greedy gives ~1; padding should fill to max_count
        base = np.zeros(128, dtype=np.float32)
        encs = [base + np.random.randn(128).astype(np.float32) * 0.0001
                for _ in range(50)]
        result = _cluster_to_max(encs, max_count=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _extract_face_encoding (via mock detector + embedder)
# ---------------------------------------------------------------------------

class TestExtractFaceEncoding:
    def _make_detection(self, conf=0.9):
        from core.detector import FaceDetection
        return FaceDetection(bbox=(10, 10, 50, 50), confidence=conf)

    def test_returns_encoding_when_detector_finds_face(self):
        mock_detector = MagicMock()
        mock_embedder = MagicMock()
        expected = np.ones(128, dtype=np.float32)

        mock_detector.detect.return_value = [self._make_detection()]
        mock_embedder.encode.return_value = expected

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _extract_face_encoding(image, mock_detector, mock_embedder,
                                        logging.getLogger("test"))
        assert result is not None
        np.testing.assert_array_equal(result, expected)

    def test_returns_none_when_no_detection(self):
        mock_detector = MagicMock()
        mock_embedder = MagicMock()
        mock_detector.detect.return_value = []

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("core.encoder.face_recognition", create=True) as mock_fr:
            mock_fr.face_encodings.return_value = []
            result = _extract_face_encoding(image, mock_detector, mock_embedder,
                                            logging.getLogger("test"))
        assert result is None

    def test_picks_highest_confidence_detection(self):
        from core.detector import FaceDetection
        mock_detector = MagicMock()
        mock_embedder = MagicMock()

        low_conf  = FaceDetection(bbox=(0, 0, 20, 20), confidence=0.4)
        high_conf = FaceDetection(bbox=(5, 5, 40, 40), confidence=0.9)
        mock_detector.detect.return_value = [low_conf, high_conf]
        mock_embedder.encode.return_value = np.ones(128, dtype=np.float32)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _extract_face_encoding(image, mock_detector, mock_embedder,
                               logging.getLogger("test"))

        # embedder.encode called with the high-confidence detection
        call_args = mock_embedder.encode.call_args
        assert call_args[0][1].confidence == 0.9

    def test_hog_fallback_when_encoder_returns_none(self):
        """When embedder.encode returns None, HOG fallback is tried."""
        mock_detector = MagicMock()
        mock_embedder = MagicMock()
        mock_detector.detect.return_value = [self._make_detection()]
        mock_embedder.encode.return_value = None  # encode fails

        fallback_enc = np.ones(128, dtype=np.float32)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # face_recognition is imported inside the function (lazy import), so patch
        # via sys.modules rather than the module attribute.
        mock_fr = MagicMock()
        mock_fr.face_encodings.return_value = [fallback_enc]
        with patch.dict("sys.modules", {"face_recognition": mock_fr}):
            result = _extract_face_encoding(image, mock_detector, mock_embedder,
                                            logging.getLogger("test"))

        assert result is not None
        np.testing.assert_array_equal(result, fallback_enc)


# ---------------------------------------------------------------------------
# rebuild_encodings
# ---------------------------------------------------------------------------

class TestRebuildEncodings:
    @pytest.fixture
    def dataset(self, tmp_path):
        """Dataset with 2 persons and 3 images each."""
        for name in ("Alice", "Bob"):
            d = tmp_path / name
            d.mkdir()
            for i in range(3):
                # Write minimal valid JPEG (1×1 pixel)
                import cv2
                img = np.zeros((50, 50, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"{name}_{i}.jpg"), img)
        return tmp_path

    def _mock_backend(self):
        """Return (mock_detector, mock_embedder) producing valid 128-d encodings."""
        from core.detector import FaceDetection
        detector = MagicMock()
        embedder = MagicMock()
        detector.detect.return_value = [FaceDetection(bbox=(5, 5, 45, 45), confidence=0.9)]
        embedder.encode.return_value = np.random.rand(128).astype(np.float32)
        return detector, embedder

    def test_creates_pkl_file(self, dataset, tmp_path):
        enc_path = str(tmp_path / "out.pkl")
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            rebuild_encodings(str(dataset), "fake_model.onnx", enc_path)
        assert os.path.exists(enc_path)

    def test_pkl_contains_encodings_and_names(self, dataset, tmp_path):
        enc_path = str(tmp_path / "out.pkl")
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            rebuild_encodings(str(dataset), "fake_model.onnx", enc_path)

        with open(enc_path, "rb") as f:
            data = pickle.load(f)
        assert "encodings" in data
        assert "names" in data
        assert len(data["encodings"]) == len(data["names"])

    def test_pkl_contains_backend_tag(self, dataset, tmp_path):
        enc_path = str(tmp_path / "out.pkl")
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            rebuild_encodings(str(dataset), "fake_model.onnx", enc_path, backend="dlib")

        with open(enc_path, "rb") as f:
            data = pickle.load(f)
        assert data["backend"] == "dlib"

    def test_encodes_both_persons(self, dataset, tmp_path):
        enc_path = str(tmp_path / "out.pkl")
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            total, persons = rebuild_encodings(str(dataset), "fake_model.onnx", enc_path)

        assert persons == 2

    def test_backs_up_existing_pkl(self, dataset, tmp_path):
        enc_path = str(tmp_path / "out.pkl")
        backup_path = str(tmp_path / "out_backup.pkl")

        # Create a "previous" pkl
        with open(enc_path, "wb") as f:
            pickle.dump({"encodings": [], "names": []}, f)

        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            rebuild_encodings(str(dataset), "fake_model.onnx", enc_path)

        assert os.path.exists(backup_path)

    def test_missing_dataset_returns_zero(self, tmp_path):
        result = rebuild_encodings("/nonexistent", "model.onnx", str(tmp_path / "out.pkl"))
        assert result == (0, 0)

    def test_empty_dataset_creates_empty_pkl(self, tmp_path):
        empty_dataset = tmp_path / "empty_dataset"
        empty_dataset.mkdir()
        enc_path = str(tmp_path / "out.pkl")

        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            total, persons = rebuild_encodings(str(empty_dataset), "model.onnx", enc_path)

        assert total == 0
        assert persons == 0


# ---------------------------------------------------------------------------
# update_person_encodings
# ---------------------------------------------------------------------------

class TestUpdatePersonEncodings:
    def _make_images(self, tmp_path, count=3) -> list:
        import cv2
        paths = []
        for i in range(count):
            p = str(tmp_path / f"img_{i}.jpg")
            cv2.imwrite(p, np.zeros((50, 50, 3), dtype=np.uint8))
            paths.append(p)
        return paths

    def _mock_backend(self):
        from core.detector import FaceDetection
        detector = MagicMock()
        embedder = MagicMock()
        detector.detect.return_value = [FaceDetection(bbox=(5, 5, 45, 45), confidence=0.9)]
        embedder.encode.return_value = np.random.rand(128).astype(np.float32)
        return detector, embedder

    def test_creates_pkl_when_no_existing(self, tmp_path):
        paths = self._make_images(tmp_path)
        enc_path = str(tmp_path / "enc.pkl")

        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            new_added, total = update_person_encodings(
                "Alice", paths, enc_path, "model.onnx"
            )

        assert os.path.exists(enc_path)
        assert new_added > 0
        assert total > 0

    def test_preserves_other_persons(self, tmp_path):
        import cv2
        # Pre-populate pkl with Bob
        bob_enc = np.ones(128, dtype=np.float32) * 0.5
        enc_path = str(tmp_path / "enc.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump({"encodings": [bob_enc], "names": ["Bob"], "backend": "dlib"}, f)

        alice_imgs = self._make_images(tmp_path, 2)
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            update_person_encodings("Alice", alice_imgs, enc_path, "model.onnx")

        with open(enc_path, "rb") as f:
            data = pickle.load(f)

        assert "Bob" in data["names"]    # Bob preserved
        assert "Alice" in data["names"]  # Alice added

    def test_no_images_returns_zero(self, tmp_path):
        enc_path = str(tmp_path / "enc.pkl")
        result = update_person_encodings("Alice", [], enc_path, "model.onnx")
        assert result == (0, 0)

    def test_backs_up_existing_pkl(self, tmp_path):
        enc_path = str(tmp_path / "enc.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump({"encodings": [], "names": [], "backend": "dlib"}, f)

        paths = self._make_images(tmp_path, 2)
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            update_person_encodings("Alice", paths, enc_path, "model.onnx")

        backup_path = str(tmp_path / "enc_backup.pkl")
        assert os.path.exists(backup_path)

    def test_total_encodings_capped_at_max(self, tmp_path):
        """Result should never exceed MAX_ENCODINGS_PER_PERSON."""
        # Pre-fill Alice with max encodings
        max_n = config.MAX_ENCODINGS_PER_PERSON
        alice_encs = [np.random.rand(128).astype(np.float32) for _ in range(max_n)]
        enc_path = str(tmp_path / "enc.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump({
                "encodings": alice_encs,
                "names": ["Alice"] * max_n,
                "backend": "dlib",
            }, f)

        paths = self._make_images(tmp_path, 10)
        with patch("core.encoder._build_backend", return_value=self._mock_backend()):
            _, total = update_person_encodings("Alice", paths, enc_path, "model.onnx")

        assert total <= max_n
