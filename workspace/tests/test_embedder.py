"""
Unit tests for core/embedder.py — FaceEmbedder interface, DlibEmbedder, SFaceEmbedder.

face_recognition (dlib) and cv2.FaceRecognizerSF are mocked so no model files
or heavy native libraries are required to run these tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.detector import FaceDetection
from core.embedder import FaceEmbedder, DlibEmbedder, SFaceEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(bbox=(10, 10, 80, 80), conf=0.9, raw_row=None) -> FaceDetection:
    row = raw_row if raw_row is not None else np.zeros(15, dtype=np.float32)
    return FaceDetection(bbox=bbox, confidence=conf, _raw_row=row)


def _black_frame(h=120, w=160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# FaceEmbedder ABC
# ---------------------------------------------------------------------------

class TestFaceEmbedderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            FaceEmbedder()  # type: ignore

    def test_concrete_subclass_works(self):
        class ConcreteEmbedder(FaceEmbedder):
            def encode(self, frame, detection, padding=15):
                return np.zeros(128, dtype=np.float32)
            def batch_distance(self, known, query):
                return np.zeros(len(known))

        emb = ConcreteEmbedder()
        frame = _black_frame()
        det = _det()
        result = emb.encode(frame, det)
        assert result.shape == (128,)

    def test_close_is_no_op_by_default(self):
        class MinimalEmbedder(FaceEmbedder):
            def encode(self, f, d, padding=15): return None
            def batch_distance(self, k, q): return np.array([])

        MinimalEmbedder().close()   # should not raise


# ---------------------------------------------------------------------------
# DlibEmbedder
# ---------------------------------------------------------------------------

class TestDlibEmbedder:
    @pytest.fixture
    def embedder(self):
        return DlibEmbedder()

    def test_encode_returns_128d_array(self, embedder):
        enc_128d = np.random.rand(128).astype(np.float32)
        with patch("face_recognition.face_encodings", return_value=[enc_128d]):
            result = embedder.encode(_black_frame(), _det())
        assert result is not None
        assert result.shape == (128,)

    def test_encode_returns_none_when_no_face_found(self, embedder):
        with patch("face_recognition.face_encodings", return_value=[]):
            result = embedder.encode(_black_frame(), _det())
        assert result is None

    def test_encode_returns_none_on_exception(self, embedder):
        with patch("face_recognition.face_encodings", side_effect=Exception("boom")):
            result = embedder.encode(_black_frame(), _det())
        assert result is None

    def test_encode_uses_bbox_as_known_location(self, embedder):
        """DlibEmbedder must pass bbox (with padding) as known_face_locations."""
        enc = np.ones(128, dtype=np.float32)
        captured_locations = []

        def fake_encodings(rgb, known_face_locations=None):
            captured_locations.append(known_face_locations)
            return [enc]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = FaceDetection(bbox=(20, 20, 60, 60), confidence=0.9)

        with patch("face_recognition.face_encodings", side_effect=fake_encodings):
            embedder.encode(frame, det, padding=10)

        assert captured_locations[0] is not None
        top, right, bottom, left = captured_locations[0][0]
        # With padding=10: top=10, right=70, bottom=70, left=10
        assert top  == 10
        assert right == 70
        assert bottom == 70
        assert left  == 10

    def test_encode_padding_clamped_to_frame(self, embedder):
        """Padding that would go out of bounds should be clamped."""
        enc = np.ones(128, dtype=np.float32)
        captured = []

        def fake_enc(rgb, known_face_locations=None):
            captured.append(known_face_locations)
            return [enc]

        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # bbox at corners, large padding → should be clamped to [0, frame_dim]
        det = FaceDetection(bbox=(0, 0, 50, 50), confidence=0.9)
        with patch("face_recognition.face_encodings", side_effect=fake_enc):
            embedder.encode(frame, det, padding=20)

        top, right, bottom, left = captured[0][0]
        assert top    >= 0
        assert left   >= 0
        assert right  <= 50
        assert bottom <= 50

    def test_batch_distance_returns_correct_shape(self, embedder):
        known = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        query = np.random.rand(128).astype(np.float32)
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with patch("face_recognition.face_distance", return_value=distances):
            result = embedder.batch_distance(known, query)

        assert result.shape == (5,)

    def test_batch_distance_delegates_to_face_recognition(self, embedder):
        known = [np.ones(128, dtype=np.float32) * i for i in range(3)]
        query = np.ones(128, dtype=np.float32)
        expected = np.array([0.0, 0.1, 0.2])

        with patch("face_recognition.face_distance", return_value=expected) as mock_fd:
            embedder.batch_distance(known, query)
            mock_fd.assert_called_once()


# ---------------------------------------------------------------------------
# SFaceEmbedder
# ---------------------------------------------------------------------------

class TestSFaceEmbedder:
    @pytest.fixture
    def embedder(self, tmp_path):
        """SFaceEmbedder with mocked cv2.FaceRecognizerSF."""
        model_path = str(tmp_path / "sface.onnx")
        Path(model_path).write_bytes(b"fake")

        mock_recognizer = MagicMock()
        aligned = np.zeros((112, 112, 3), dtype=np.uint8)
        feature = np.ones((1, 128), dtype=np.float32)
        mock_recognizer.alignCrop.return_value = aligned
        mock_recognizer.feature.return_value = feature

        with patch("cv2.FaceRecognizerSF.create", return_value=mock_recognizer):
            emb = SFaceEmbedder(model_path=model_path)

        return emb, mock_recognizer

    def test_encode_returns_none_without_raw_row(self, tmp_path):
        """SFaceEmbedder requires _raw_row from YuNet; should return None without it."""
        model_path = str(tmp_path / "sface.onnx")
        Path(model_path).write_bytes(b"fake")
        with patch("cv2.FaceRecognizerSF.create", return_value=MagicMock()):
            emb = SFaceEmbedder(model_path=model_path)

        det_no_raw = FaceDetection(bbox=(10, 10, 80, 80), confidence=0.9, _raw_row=None)
        result = emb.encode(_black_frame(), det_no_raw)
        assert result is None

    def test_encode_returns_128d_when_raw_row_present(self, embedder):
        emb, _ = embedder
        det = _det()  # has _raw_row
        result = emb.encode(_black_frame(), det)
        assert result is not None
        assert result.shape == (128,)

    def test_encode_calls_align_crop_then_feature(self, embedder):
        emb, mock_rec = embedder
        det = _det()
        emb.encode(_black_frame(), det)
        mock_rec.alignCrop.assert_called_once()
        mock_rec.feature.assert_called_once()

    def test_encode_returns_none_on_exception(self, tmp_path):
        model_path = str(tmp_path / "sface.onnx")
        Path(model_path).write_bytes(b"fake")
        mock_rec = MagicMock()
        mock_rec.alignCrop.side_effect = Exception("model error")

        with patch("cv2.FaceRecognizerSF.create", return_value=mock_rec):
            emb = SFaceEmbedder(model_path=model_path)

        result = emb.encode(_black_frame(), _det())
        assert result is None

    def test_batch_distance_returns_cosine_in_0_1(self, embedder):
        """cosine distance = 1 - cosine_score should be in [0, 1]."""
        import cv2
        emb, mock_rec = embedder
        # Simulate cosine_score close to 1.0 (very similar)
        mock_rec.match.return_value = 0.95

        known = [np.random.rand(128).astype(np.float32) for _ in range(3)]
        query = np.random.rand(128).astype(np.float32)
        result = emb.batch_distance(known, query)

        assert result.shape == (3,)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_batch_distance_zero_when_identical(self, embedder):
        """Same vector → cosine_score = 1.0 → distance = 0.0."""
        emb, mock_rec = embedder
        mock_rec.match.return_value = 1.0  # perfect match

        query = np.ones(128, dtype=np.float32)
        result = emb.batch_distance([query], query)
        assert result[0] == pytest.approx(0.0)

    def test_batch_distance_shape_matches_known_count(self, embedder):
        emb, mock_rec = embedder
        mock_rec.match.return_value = 0.5
        known = [np.random.rand(128).astype(np.float32) for _ in range(7)]
        query = np.random.rand(128).astype(np.float32)
        result = emb.batch_distance(known, query)
        assert result.shape == (7,)

    def test_close_does_not_raise(self, embedder):
        emb, _ = embedder
        emb.close()   # should be a no-op
