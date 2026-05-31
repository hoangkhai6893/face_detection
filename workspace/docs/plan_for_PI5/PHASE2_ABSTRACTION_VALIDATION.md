# Phase 2 — Abstraction Layer & Validation Gate

> **Mục tiêu:** Tách interface detector/embedder khỏi business logic + xây A/B harness để quyết định an toàn trước khi migrate model.
> **Quan trọng:** Phase này KHÔNG thay đổi model nào. Kết quả nhận diện phải **bit-for-bit giống hệt** trước và sau Phase 2.
> **Gate:** validation harness phải xác nhận accuracy(YuNet+SFace) ≥ accuracy(YOLO+dlib) mới tiến hành Phase 3.

---

## 2.1 Thiết Kế Interface

### `src/core/detector.py` (NEW)

```python
"""
FaceDetector — abstract interface cho face detection.
Backends: YoloDetector (hiện tại), YuNetDetector (target Phase 3).
"""
from __future__ import annotations
import abc
from typing import List, Optional, Tuple
import numpy as np


class FaceDetection:
    """Kết quả detect 1 khuôn mặt. Bất biến."""
    __slots__ = ("bbox", "confidence", "landmarks", "_raw_row")

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],   # (x1, y1, x2, y2) trong không gian frame gốc
        confidence: float,
        landmarks: Optional[np.ndarray] = None,  # shape (5, 2) [x,y] per landmark, hoặc None
        _raw_row: Optional[np.ndarray] = None,   # raw output row để YuNet alignCrop dùng
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        self._raw_row = _raw_row


class FaceDetector(abc.ABC):
    """Interface: nhận BGR frame, trả List[FaceDetection]."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Args:
            frame: BGR image (numpy array, uint8)
        Returns:
            List FaceDetection đã lọc bởi detection_confidence
        """
        ...

    def warmup(self, size: Tuple[int, int] = (64, 64)) -> None:
        """Chạy 1 inference dummy để JIT compile. Gọi sau __init__."""
        dummy = np.zeros((*size, 3), dtype=np.uint8)
        self.detect(dummy)

    def close(self) -> None:
        """Giải phóng resources nếu cần."""
        pass
```

### `src/core/embedder.py` (NEW)

```python
"""
FaceEmbedder — abstract interface cho face embedding & matching.

Distance convention (QUAN TRỌNG):
  Cả hai backend đều trả distance trong [0, ∞):
    0.0 = hoàn toàn giống nhau
    cao hơn = khác nhau hơn
  → TOLERANCE và CONFUSION_MARGIN trong config dùng cùng semantics cho cả hai.

DlibEmbedder:  Euclidean distance (face_recognition.face_distance)
SFaceEmbedder: 1 - cosine_score (FR_COSINE), normalize về [0, 1]
"""
from __future__ import annotations
import abc
from typing import List, Optional
import numpy as np
from .detector import FaceDetection


class FaceEmbedder(abc.ABC):

    @abc.abstractmethod
    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 15,
    ) -> Optional[np.ndarray]:
        """
        Trả 128-d embedding vector, hoặc None nếu không encode được.
        frame_bgr: BGR full frame (KHÔNG phải crop)
        detection: vị trí face (bbox + optional landmarks)
        padding: pixel thêm quanh bbox
        """
        ...

    @abc.abstractmethod
    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        """
        Trả array shape (N,) distances giữa query và từng known encoding.
        Semantics: 0 = giống nhau, cao = khác nhau.
        """
        ...

    def close(self) -> None:
        pass
```

---

## 2.2 Backend Hiện Tại — Wrap vào Interface

### `src/core/detector.py` — thêm `YoloDetector`

```python
class YoloDetector(FaceDetector):
    """Wrapper quanh ultralytics YOLO. Giữ nguyên logic từ recognizer.py."""

    def __init__(
        self,
        model_path: str,
        detection_confidence: float = 0.3,
        input_width: int = 416,
    ) -> None:
        from ultralytics import YOLO as _YOLO
        self._model = _YOLO(model_path)
        self._conf = detection_confidence
        self._input_width = input_width
        self.warmup()

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        # Resize nếu cần
        if self._input_width and self._input_width < w:
            scale = self._input_width / w
            small = cv2.resize(frame, (self._input_width, int(h * scale)))
            inv = 1.0 / scale
        else:
            small, inv = frame, 1.0

        results = self._model(small, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self._conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Scale trở lại không gian frame gốc
                bbox = (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv))
                detections.append(FaceDetection(bbox=bbox, confidence=conf))
        return detections
```

### `src/core/embedder.py` — thêm `DlibEmbedder`

```python
class DlibEmbedder(FaceEmbedder):
    """Wrapper quanh face_recognition (dlib ResNet). Giữ nguyên logic từ recognizer.py."""

    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 15,
    ) -> Optional[np.ndarray]:
        import face_recognition
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        top    = max(0, y1 - padding)
        right  = min(w, x2 + padding)
        bottom = min(h, y2 + padding)
        left   = max(0, x1 - padding)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            encodings = face_recognition.face_encodings(
                rgb, known_face_locations=[(top, right, bottom, left)]
            )
            return encodings[0] if encodings else None
        except Exception:
            return None

    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        import face_recognition
        return face_recognition.face_distance(known_encodings, query_encoding)
```

---

## 2.3 Refactor `recognizer.py` để Dùng Interface

**Mục tiêu:** thay `self.yolo_model`, `face_recognition.*` bằng `self.detector`, `self.embedder`.
**Ràng buộc:** kết quả nhận diện **phải giống hệt** với trước khi refactor.

### `__init__` thay đổi:

```python
# TRƯỚC:
self.yolo_model = YOLO(model_path)
self.yolo_model(np.zeros(...), verbose=False)   # warmup

# SAU:
from core.detector import YoloDetector
from core.embedder import DlibEmbedder
self.detector = YoloDetector(
    model_path=model_path,
    detection_confidence=detection_confidence,
    input_width=yolo_input_width,
)
self.embedder = DlibEmbedder()
```

### `run_recognition()` main loop — thay đổi:

```python
# TRƯỚC (dòng 495-509):
if yolo_scale < 1.0:
    small_frame = cv2.resize(frame, ...)
    small_detections = self.detect_faces_yolo(small_frame)
    face_detections = [rescale...]
else:
    face_detections = self.detect_faces_yolo(frame)

# SAU — YoloDetector đã handle resize + rescale nội bộ:
face_detections_raw = self.detector.detect(frame)
# convert FaceDetection → tuple (x1,y1,x2,y2,conf) cho tương thích code phía sau
face_detections = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence)
                   for d in face_detections_raw]
_raw_detections = face_detections_raw   # giữ cho IOU cache lookup
```

### `recognize_face_in_region()` — thay đổi:

```python
# TRƯỚC: gọi trực tiếp face_recognition.face_encodings() và face_distance()

# SAU: dùng self.embedder
face_encoding = self.embedder.encode(frame_bgr, detection, padding=self.padding)
if face_encoding is None:
    return "No Face"
distances = self.embedder.batch_distance(self.known_face_encodings, face_encoding)
# ... voting logic giữ nguyên
```

**Lưu ý:** `frame_bgr` cần truyền xuống `recognize_face_in_region()`. Hiện tại đang truyền `rgb_frame` (đã convert). Điều chỉnh: convert BGR→RGB trong DlibEmbedder nội bộ, embedder nhận BGR.

### `_fallback_extract_encoding()` — thay bằng DlibEmbedder fallback:

```python
def _fallback_extract_encoding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """HOG full-image fallback khi detector miss."""
    t0 = time.time()
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        import face_recognition
        encs = face_recognition.face_encodings(rgb)
        elapsed_ms = (time.time() - t0) * 1000
        if encs:
            self.logger.warning("HOG fallback triggered — %.0fms", elapsed_ms)
            self._hog_fallback_count += 1
            return encs[0]
        return None
    except Exception:
        return None
```

---

## 2.4 Validation Harness — `scripts/validate_backends.py` (NEW)

**Mục tiêu:** chạy cả 2 backend trên cùng held-out test data, xuất báo cáo, quyết định có migrate hay không.

### Cách tạo test set

```bash
# 1. Từ mỗi người trong gia đình: chọn 10-20 ảnh/video clip KHÔNG nằm trong training
# 2. Cấu trúc:
#    test_data/
#      Khai/     ← 10-20 ảnh hoặc 1 clip mp4
#      MaiAnh/
#      Unknown/  ← ảnh người lạ thật (không ai trong gia đình)

# 3. Gán nhãn:
#    test_labels.csv:
#      filepath, true_label
#      test_data/Khai/img001.jpg, Khai
#      test_data/Unknown/stranger1.jpg, Unknown
```

### Interface script:

```python
# scripts/validate_backends.py
"""
So sánh 2 backend: YOLO+Dlib vs YuNet+SFace trên held-out test data.
Kết quả: báo cáo accuracy + speed, quyết định migration.

Usage:
  python scripts/validate_backends.py \
    --test-dir test_data/ \
    --labels test_labels.csv \
    --encodings model/face_encodings_hybrid.pkl \
    --report validation_report.json
"""

def run_validation(
    test_dir: str,
    labels_csv: str,
    encodings_pkl: str,
    report_path: str = "validation_report.json",
) -> dict:
    """
    Returns dict with keys:
      backend_a: {accuracy, false_unknown_rate, confusion_rate, avg_ms_per_face}
      backend_b: {accuracy, false_unknown_rate, confusion_rate, avg_ms_per_face}
      decision: "PASS" | "FAIL" | "NEEDS_TUNING"
    """
    ...
```

### Metrics cần đo:

| Metric | Định nghĩa | Threshold để PASS |
|---|---|---|
| `accuracy` | % ảnh nhận đúng person label | B ≥ A |
| `false_unknown_rate` | % ảnh người nhà bị nhận là "Unknown" | B ≤ A + 0.05 (5% slack) |
| `confusion_rate` | % nhầm người nhà này sang người nhà khác | B ≤ A (zero tolerance) |
| `avg_ms_per_face` | Thời gian encode + match 1 mặt | B ≤ A * 1.5 (chấp nhận B chậm hơn 50% nếu accuracy tốt) |

### Logic gate:

```python
def decide_migration(a: dict, b: dict) -> str:
    """
    PASS     → proceed với Phase 3
    FAIL     → giữ stack A, không migrate
    NEEDS_TUNING → B accuracy kém hơn nhưng gần (< 5%) → thử tune threshold trước
    """
    if b["confusion_rate"] > a["confusion_rate"]:
        return "FAIL"   # nhầm người nhà → không chấp nhận
    if b["accuracy"] >= a["accuracy"] and b["false_unknown_rate"] <= a["false_unknown_rate"] + 0.05:
        return "PASS"
    if b["accuracy"] >= a["accuracy"] - 0.05:
        return "NEEDS_TUNING"
    return "FAIL"
```

---

## 2.5 Kiểm Tra Sau Phase 2

```bash
# 1. Kết quả recognition phải giống hệt với stack YOLO+dlib trực tiếp
# Chạy cùng 1 video clip qua 2 code path:
#   a) code mới (qua YoloDetector + DlibEmbedder wrapper)
#   b) code cũ (git stash show)
# So sánh output: tên nhận được phải 100% giống nhau.

# 2. Test suite pass
pytest tests/ -v
# Cập nhật tests nếu cần mock YoloDetector thay vì YOLO trực tiếp

# 3. Chạy harness với 2 backend cùng encodings
python scripts/validate_backends.py \
  --test-dir test_data/ \
  --labels test_labels.csv \
  --encodings model/face_encodings_hybrid.pkl
# Output: validation_report.json
# Gate: nếu decision = "PASS" → tiến hành Phase 3
```

---

## Rollback Phase 2

Phase 2 chỉ **thêm code** (detector.py, embedder.py, validate_backends.py) và **refactor** recognizer.py. Không xóa dependency nào.

```bash
git revert <phase2-commit>
# Hoặc chạy lại code cũ trong cùng repo (nhánh)
```

---

## Checklist Trước Khi Sang Phase 3

- [ ] `pytest tests/ -v` → tất cả pass
- [ ] `python scripts/validate_backends.py ...` → `decision: PASS`
- [ ] Không có regression nào trong kết quả nhận diện với người nhà
- [ ] Độ trễ nhận diện trên Pi5 được đo và ghi lại (baseline để so sánh sau Phase 3)
- [ ] Backup `model/face_encodings_hybrid.pkl` (sẽ rebuild trong Phase 3)
