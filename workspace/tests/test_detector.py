"""
Unit tests for core/detector.py — FaceDetection, FaceDetector interface, YoloDetector.

YoloDetector is tested with a mocked ultralytics.YOLO model so no model files
are required. YuNetDetector structure tests verify the class can be imported
and has the correct interface without requiring actual model files.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.detector import FaceDetection, FaceDetector, YoloDetector


# ---------------------------------------------------------------------------
# FaceDetection dataclass
# ---------------------------------------------------------------------------

class TestFaceDetection:
    def test_basic_construction(self):
        det = FaceDetection(bbox=(10, 20, 100, 120), confidence=0.85)
        assert det.bbox == (10, 20, 100, 120)
        assert det.confidence == pytest.approx(0.85)
        assert det.landmarks is None
        assert det._raw_row is None

    def test_with_landmarks(self):
        landmarks = np.array([[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]],
                             dtype=np.float32)
        det = FaceDetection(bbox=(0, 0, 50, 50), confidence=0.9, landmarks=landmarks)
        assert det.landmarks.shape == (5, 2)

    def test_with_raw_row(self):
        raw = np.zeros(15, dtype=np.float32)
        det = FaceDetection(bbox=(0, 0, 50, 50), confidence=0.7, _raw_row=raw)
        assert det._raw_row is not None
        assert len(det._raw_row) == 15

    def test_bbox_is_xyxy_tuple(self):
        """Bbox must always be (x1, y1, x2, y2)."""
        det = FaceDetection(bbox=(5, 10, 80, 90), confidence=0.8)
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2
        assert y1 < y2


# ---------------------------------------------------------------------------
# FaceDetector ABC
# ---------------------------------------------------------------------------

class TestFaceDetectorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            FaceDetector()  # type: ignore

    def test_concrete_subclass_works(self):
        class ConcreteDetector(FaceDetector):
            def detect(self, frame):
                return []

        det = ConcreteDetector()
        assert det.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []

    def test_warmup_calls_detect(self):
        class TrackingDetector(FaceDetector):
            def __init__(self):
                self.called_with_shapes = []

            def detect(self, frame):
                self.called_with_shapes.append(frame.shape)
                return []

        det = TrackingDetector()
        det.warmup(size=(32, 32))
        assert len(det.called_with_shapes) == 1
        assert det.called_with_shapes[0] == (32, 32, 3)

    def test_close_is_no_op_by_default(self):
        class MinimalDetector(FaceDetector):
            def detect(self, frame):
                return []

        MinimalDetector().close()   # should not raise


# ---------------------------------------------------------------------------
# YoloDetector (mocked ultralytics.YOLO)
# ---------------------------------------------------------------------------

def _make_yolo_result(boxes_data: list):
    """Build a fake ultralytics result with given boxes.

    boxes_data: list of (x1, y1, x2, y2, conf)
    """
    result = MagicMock()
    boxes = []
    for x1, y1, x2, y2, conf in boxes_data:
        box = MagicMock()
        box.conf = [conf]
        xyxy = MagicMock()
        xyxy.cpu.return_value.numpy.return_value = [x1, y1, x2, y2]
        box.xyxy = [xyxy]
        boxes.append(box)
    result.boxes = boxes if boxes else None
    return result


def _make_yolo_model(*boxes_data):
    """Create a mock YOLO model that returns the given detections."""
    model = MagicMock()
    model.return_value = [_make_yolo_result(list(boxes_data))]
    return model


@pytest.fixture
def patched_yolo_detector(tmp_path):
    """YoloDetector with ultralytics.YOLO patched — no real model file needed."""
    fake_model_path = str(tmp_path / "fake.onnx")
    Path(fake_model_path).write_bytes(b"fake")

    with patch("core.detector.cv2") as _:  # cv2 import in detector
        with patch("ultralytics.YOLO", return_value=MagicMock()) as mock_yolo_cls:
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            # Patch the import inside YoloDetector.__init__
            with patch("core.detector.YoloDetector._YOLO_CLS", create=True):
                # Use direct injection approach instead
                pass

    # Simpler: directly patch via __init__ by monkey-patching the YOLO import
    detector = YoloDetector.__new__(YoloDetector)
    detector._conf = 0.3
    detector._input_width = 416
    detector._model = MagicMock()
    return detector


class TestYoloDetector:
    def _make_detector(self, conf=0.3, input_width=416):
        """Build YoloDetector with a mocked YOLO model (no file needed)."""
        with patch("ultralytics.YOLO") as mock_cls:
            mock_model = MagicMock()
            mock_model.return_value = []   # warmup call
            mock_cls.return_value = mock_model

            with patch("core.detector.YoloDetector.warmup"):   # skip warmup
                from core.detector import YoloDetector as _YD
                det = _YD.__new__(_YD)
                det._conf = conf
                det._input_width = input_width
                det._model = mock_model
                return det

    def test_returns_empty_on_no_detections(self):
        det = self._make_detector()
        det._model.return_value = [_make_yolo_result([])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert result == []

    def test_returns_face_detections(self):
        det = self._make_detector()
        det._model.return_value = [_make_yolo_result([(100, 50, 200, 150, 0.9)])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 1
        assert isinstance(result[0], FaceDetection)
        assert result[0].confidence == pytest.approx(0.9)

    def test_filters_by_confidence(self):
        det = self._make_detector(conf=0.5)
        # Two detections: one above, one below threshold
        det._model.return_value = [_make_yolo_result([
            (10, 10, 50, 50, 0.7),   # above → kept
            (60, 60, 80, 80, 0.3),   # below → filtered
        ])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.7)

    def test_multiple_detections_returned(self):
        det = self._make_detector(conf=0.3)
        det._model.return_value = [_make_yolo_result([
            (10, 10, 50, 50, 0.9),
            (100, 100, 200, 200, 0.8),
            (300, 50, 400, 150, 0.6),
        ])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 3

    def test_bbox_scaled_to_original_frame_when_resized(self):
        """When input_width causes downscaling, bbox must be in original frame space."""
        det = self._make_detector(conf=0.3, input_width=320)
        # Original frame is 640×480; input_width=320 → scale=0.5, inv=2.0
        # Detection at (50, 25, 150, 75) in small frame → (100, 50, 300, 150) in original
        det._model.return_value = [_make_yolo_result([(50, 25, 150, 75, 0.85)])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        # Allow small rounding differences (±2 px)
        assert abs(x1 - 100) <= 2
        assert abs(y1 - 50) <= 2
        assert abs(x2 - 300) <= 2
        assert abs(y2 - 150) <= 2

    def test_no_resize_when_frame_smaller_than_input_width(self):
        """Small frames should not be upscaled; bbox stays in original coords."""
        det = self._make_detector(conf=0.3, input_width=640)
        det._model.return_value = [_make_yolo_result([(10, 10, 50, 50, 0.9)])]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)   # 320 < 640
        result = det.detect(frame)
        assert len(result) == 1
        assert result[0].bbox == (10, 10, 50, 50)

    def test_bbox_clamped_to_frame_bounds(self):
        """Negative or out-of-bounds coords should be clamped."""
        det = self._make_detector(conf=0.3, input_width=0)  # no resize
        det._model.return_value = [_make_yolo_result([(-10, -5, 700, 500, 0.9)])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_detect_returns_face_detection_type(self):
        det = self._make_detector()
        det._model.return_value = [_make_yolo_result([(10, 10, 50, 50, 0.9)])]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert all(isinstance(d, FaceDetection) for d in result)


# ---------------------------------------------------------------------------
# YuNetDetector — import and interface check only (model file not required)
# ---------------------------------------------------------------------------

class TestYuNetDetectorInterface:
    def test_can_import_yunet_detector(self):
        from core.detector import YuNetDetector
        assert YuNetDetector is not None

    def test_yunet_is_subclass_of_face_detector(self):
        from core.detector import YuNetDetector
        assert issubclass(YuNetDetector, FaceDetector)

    def test_detect_returns_empty_when_cv2_detects_nothing(self, monkeypatch):
        """If cv2.FaceDetectorYN.detect() returns (_, None), result should be []."""
        from core.detector import YuNetDetector
        import cv2

        mock_cv2_detector = MagicMock()
        mock_cv2_detector.detect.return_value = (None, None)
        mock_cv2_detector.setInputSize = MagicMock()

        with patch("cv2.FaceDetectorYN.create", return_value=mock_cv2_detector):
            det = YuNetDetector.__new__(YuNetDetector)
            det._conf = 0.6
            det._input_size = (320, 240)
            det._detector = mock_cv2_detector

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert result == []
