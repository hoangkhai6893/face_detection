# Phase 3 — Migration YuNet + SFace (OpenCV-Native Stack)

> **Điều kiện tiên quyết:** Phase 2 đã xong + validation gate PASS.
> **Mục tiêu:** Thay YOLO+dlib bằng YuNet+SFace. Kết thúc: bỏ được dlib, face_recognition, ultralytics.
> **Rủi ro:** TRUNG BÌNH — rebuild toàn bộ encodings + retune thresholds. Backup .pkl trước khi bắt đầu.

---

## 3.0 Chuẩn Bị

### Backup encodings hiện tại

```bash
cp model/face_encodings_hybrid.pkl model/face_encodings_dlib_backup_$(date +%Y%m%d).pkl
```

### Download model từ OpenCV Zoo

```bash
OPENCV_ZOO_RAW=https://raw.githubusercontent.com/opencv/opencv_zoo/main/models
mkdir -p src/yolo model

# YuNet detector (INT8 ONNX — nhanh nhất trên ARM)
wget "$OPENCV_ZOO_RAW/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx" \
     -O src/yolo/yunet_int8.onnx

# SFace embedder (INT8 ONNX)
wget "$OPENCV_ZOO_RAW/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx" \
     -O model/sface_int8.onnx

# Verify
python -c "import cv2; d=cv2.FaceDetectorYN.create('src/yolo/yunet_int8.onnx','',( 320,240)); print('YuNet OK')"
python -c "import cv2; r=cv2.FaceRecognizerSF.create('model/sface_int8.onnx',''); print('SFace OK')"
```

---

## 3.1 YuNetDetector

Thêm vào `src/core/detector.py`:

```python
class YuNetDetector(FaceDetector):
    """
    Face detector dùng YuNet (cv2.FaceDetectorYN).
    Trả 5 landmarks cho SFaceEmbedder.alignCrop().

    YuNet output format (N×15 matrix):
      col 0-3:  x, y, w, h  (top-left + size, KHÔNG phải x1y1x2y2)
      col 4-13: 5 landmarks (x,y) × 5
      col 14:   confidence score
    """

    def __init__(
        self,
        model_path: str,
        detection_confidence: float = 0.6,   # SFace hoạt động tốt với conf cao hơn
        input_size: Tuple[int, int] = (320, 240),  # (width, height) — scale frame về đây
        nms_threshold: float = 0.3,
    ) -> None:
        self._conf = detection_confidence
        self._input_size = input_size
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=detection_confidence,
            nms_threshold=nms_threshold,
            top_k=5,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        orig_h, orig_w = frame.shape[:2]
        iw, ih = self._input_size

        # Resize về input size để detect
        resized = cv2.resize(frame, (iw, ih))
        self._detector.setInputSize((iw, ih))

        _, faces = self._detector.detect(resized)
        if faces is None:
            return []

        # Scale factor trở lại frame gốc
        sx = orig_w / iw
        sy = orig_h / ih

        detections = []
        for row in faces:
            x, y, w, h = row[0], row[1], row[2], row[3]
            conf = float(row[14])
            if conf < self._conf:
                continue

            # Convert YuNet (x,y,w,h) → (x1,y1,x2,y2) trong không gian gốc
            x1 = max(0, int(x * sx))
            y1 = max(0, int(y * sy))
            x2 = min(orig_w, int((x + w) * sx))
            y2 = min(orig_h, int((y + h) * sy))

            # Scale landmarks
            landmarks = np.array([
                [row[4 + i*2] * sx, row[4 + i*2 + 1] * sy]
                for i in range(5)
            ], dtype=np.float32)  # shape (5, 2)

            # Raw row (scaled về gốc) cho alignCrop
            raw_row = row.copy()
            raw_row[0] = x * sx;  raw_row[1] = y * sy
            raw_row[2] = w * sx;  raw_row[3] = h * sy
            for i in range(5):
                raw_row[4 + i*2]     = row[4 + i*2] * sx
                raw_row[4 + i*2 + 1] = row[4 + i*2 + 1] * sy

            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                landmarks=landmarks,
                _raw_row=raw_row,
            ))

        return detections
```

---

## 3.2 SFaceEmbedder

Thêm vào `src/core/embedder.py`:

```python
class SFaceEmbedder(FaceEmbedder):
    """
    Face embedder dùng SFace (cv2.FaceRecognizerSF) với alignCrop.

    PHẢI pair với YuNetDetector vì cần FaceDetection.landmarks (5 points).
    encode() → 128-d L2-normalized feature vector.
    batch_distance() → 1 - cosine_score ∈ [0, 1] (0=same, 1=different).

    Threshold tham chiếu (official):
      cosine_score >= 0.363 → same person
      distance = 1 - cosine_score → 0.637 threshold
      → Config TOLERANCE=0.63 là điểm khởi đầu, retune trong Phase 4.
    """
    # Kích thước input SFace yêu cầu sau alignCrop
    _ALIGNED_SIZE = (112, 112)

    def __init__(self, model_path: str) -> None:
        self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")

    def encode(
        self,
        frame_bgr: np.ndarray,
        detection: FaceDetection,
        padding: int = 0,   # SFace dùng alignCrop nên không cần padding thủ công
    ) -> Optional[np.ndarray]:
        """
        Dùng landmarks để alignCrop rồi extract feature.
        _raw_row phải có sẵn trong detection (từ YuNetDetector).
        """
        if detection._raw_row is None:
            return None   # YuNet detection bắt buộc
        try:
            aligned = self._recognizer.alignCrop(frame_bgr, detection._raw_row)
            feature = self._recognizer.feature(aligned)   # shape (1, 128)
            return feature.flatten()                       # shape (128,)
        except Exception:
            return None

    def batch_distance(
        self,
        known_encodings: List[np.ndarray],
        query_encoding: np.ndarray,
    ) -> np.ndarray:
        """
        Tính cosine distance = 1 - cosine_score cho mỗi known encoding.
        Semantics: 0 = same person, 1 = completely different.
        """
        distances = np.empty(len(known_encodings), dtype=np.float32)
        q = query_encoding.reshape(1, -1).astype(np.float32)
        for i, known in enumerate(known_encodings):
            k = known.reshape(1, -1).astype(np.float32)
            score = self._recognizer.match(k, q, cv2.FaceRecognizerSF.FR_COSINE)
            distances[i] = 1.0 - float(score)   # convert sang [0=same, 1=diff]
        return distances

    def close(self) -> None:
        pass   # cv2 không có explicit close
```

---

## 3.3 Matching Logic — Thay Euclidean → Cosine Distance

**Tin tốt:** không cần sửa business logic (voting, margin check) trong `recognizer.py` vì `batch_distance()` đã normalize về cùng semantics (lower = more similar).

**Chỉ cần thay đổi config thresholds:**

### `src/config.py` — thêm section mới:

```python
# --- Embedding backend ---
# "dlib"  = face_recognition (Euclidean, TOLERANCE≈0.5)
# "sface" = SFace (cosine distance = 1-score, TOLERANCE≈0.63)
# Đổi backend: cần rebuild encodings (khác embedding space)
FACE_BACKEND: str = os.environ.get("FACE_BACKEND", "sface")  # "dlib" để rollback

# --- Model paths (Phase 3) ---
YUNET_MODEL_PATH  = os.path.join(_SRC_DIR, "yolo", "yunet_int8.onnx")
SFACE_MODEL_PATH  = os.path.join(_WORKSPACE_DIR, "model", "sface_int8.onnx")

# --- Thresholds cho SFace cosine distance (retune trong Phase 4) ---
# Distance = 1 - cosine_score. Official threshold: 0.637 (= 1 - 0.363)
# Đặt conservative hơn 1 chút (0.60) để bắt đầu, đo rồi tune.
TOLERANCE_SFACE         = float(os.environ.get("TOLERANCE_SFACE",         "0.60"))
CONFUSION_MARGIN_SFACE  = float(os.environ.get("CONFUSION_MARGIN_SFACE",  "0.12"))
```

### `recognizer.py` `__init__` — chọn backend:

```python
from core.detector import YoloDetector, YuNetDetector
from core.embedder import DlibEmbedder, SFaceEmbedder

# Chọn backend theo config
if config.FACE_BACKEND == "sface":
    self.detector = YuNetDetector(config.YUNET_MODEL_PATH, detection_confidence)
    self.embedder = SFaceEmbedder(config.SFACE_MODEL_PATH)
    self.tolerance       = config.TOLERANCE_SFACE
    self.confusion_margin = config.CONFUSION_MARGIN_SFACE
else:  # dlib (fallback / rollback)
    self.detector = YoloDetector(model_path, detection_confidence, yolo_input_width)
    self.embedder = DlibEmbedder()
    self.tolerance       = config.TOLERANCE
    self.confusion_margin = config.CONFUSION_MARGIN
```

---

## 3.4 Cập Nhật `encoder.py` — Rebuild với SFace

`_extract_face_encoding()` và `rebuild_encodings()` phải dùng YuNet+SFace thay vì YOLO+dlib.

### Thay đổi `src/core/encoder.py`:

```python
# Thêm import
from core.detector import YuNetDetector, FaceDetection
from core.embedder import SFaceEmbedder

def _build_backend(backend: str = config.FACE_BACKEND):
    """Tạo detector+embedder pair theo backend config."""
    if backend == "sface":
        return (
            YuNetDetector(config.YUNET_MODEL_PATH, detection_confidence=0.4),
            SFaceEmbedder(config.SFACE_MODEL_PATH),
        )
    else:
        from ultralytics import YOLO
        import face_recognition as _fr
        from core.detector import YoloDetector
        from core.embedder import DlibEmbedder
        return YoloDetector(config.MODEL_PATH), DlibEmbedder()


def _extract_face_encoding(
    image: np.ndarray,
    detector: "FaceDetector",
    embedder: "FaceEmbedder",
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Detect face với detector, encode với embedder.
    Trả None nếu không detect được mặt.
    """
    detections = detector.detect(image)
    if not detections:
        return None

    # Dùng detection có confidence cao nhất
    best = max(detections, key=lambda d: d.confidence)
    return embedder.encode(image, best)


def rebuild_encodings(
    dataset_path: str = config.DATASET_PATH,
    output_path: Optional[str] = None,
    backend: str = config.FACE_BACKEND,
    logger: Optional[logging.Logger] = None,
) -> int:
    ...
    detector, embedder = _build_backend(backend)
    ...
    # Phần còn lại giữ nguyên (loop qua người, encode, cluster, save)
```

---

## 3.5 Cập Nhật `frame_extractor.py`

`VideoFrameExtractor` dùng YOLO để detect trong quá trình training data collection.

### Thay đổi:

```python
# Thay YOLO bằng YuNetDetector trong __init__:
from core.detector import YuNetDetector, YoloDetector

if config.FACE_BACKEND == "sface":
    self._detector = YuNetDetector(
        config.YUNET_MODEL_PATH,
        detection_confidence=config.FRAME_DETECT_CONF,
    )
else:
    self._detector = YoloDetector(
        config.MODEL_PATH,
        detection_confidence=config.FRAME_DETECT_CONF,
    )
```

**Fix bug detect_conf (P3.3):**

```python
# TRƯỚC (frame_extractor.py dòng ~130, 374):
detection_conf=0.0   # bug: confidence thật bị bỏ qua

# SAU:
best_detection = max(self._detector.detect(image), key=lambda d: d.confidence, default=None)
if best_detection:
    detection_conf = best_detection.confidence   # giữ conf thật
    bbox = best_detection.bbox
```

---

## 3.6 Rebuild Encodings

```bash
# Backup dlib encodings
cp model/face_encodings_hybrid.pkl model/face_encodings_dlib_backup_$(date +%Y%m%d).pkl

# Rebuild với SFace
FACE_BACKEND=sface python -c "
from core.encoder import rebuild_encodings
import logging
logging.basicConfig(level=logging.INFO)
rebuild_encodings()
"

# Kiểm tra
python -c "
import pickle
data = pickle.load(open('model/face_encodings_hybrid.pkl','rb'))
persons = set(data['names'])
print(f'Persons: {persons}')
print(f'Total encodings: {len(data[\"encodings\"])}')
for p in persons:
    n = sum(1 for name in data['names'] if name == p)
    print(f'  {p}: {n} encodings')
"
```

---

## 3.7 Bỏ Dependencies Cũ

**Chỉ bỏ sau khi xác nhận stack mới hoạt động ổn định (pass Phase 4 validation).**

### Kiểm tra không còn import nào dùng deps cũ:

```bash
grep -rn "from ultralytics\|import ultralytics\|from face_recognition\|import face_recognition\|import dlib" src/
# Kỳ vọng: chỉ còn trong DlibEmbedder và YoloDetector (lazy import trong class)
```

### Khi sẵn sàng bỏ hoàn toàn:

```bash
pip uninstall dlib face-recognition ultralytics -y
# requirements.txt: bỏ 3 dòng trên, đảm bảo chỉ còn opencv-python-headless numpy
```

**Rollback nhanh nếu cần:**
```bash
# Chuyển về dlib trong .env:
FACE_BACKEND=dlib
# Reinstall:
pip install dlib face-recognition ultralytics onnxruntime
# Dùng lại backup encodings:
cp model/face_encodings_dlib_backup_*.pkl model/face_encodings_hybrid.pkl
```

---

## 3.8 Update Tests

Các tests cần cập nhật khi dùng SFace:

**`tests/conftest.py`:**
```python
# Thêm fixtures cho YuNet+SFace
@pytest.fixture
def mock_yunet_detector():
    """Mock YuNetDetector trả fixed FaceDetection với landmarks."""
    from unittest.mock import MagicMock, patch
    from core.detector import FaceDetection
    import numpy as np
    det = MagicMock()
    det.detect.return_value = [
        FaceDetection(
            bbox=(50, 50, 150, 150),
            confidence=0.9,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            _raw_row=np.zeros(15, dtype=np.float32),
        )
    ]
    return det

@pytest.fixture
def mock_sface_embedder():
    """Mock SFaceEmbedder trả fixed 128-d encoding."""
    from unittest.mock import MagicMock
    import numpy as np
    emb = MagicMock()
    emb.encode.return_value = np.random.rand(128).astype(np.float32)
    emb.batch_distance.return_value = np.array([0.3, 0.7, 0.8])
    return emb
```

**Thresholds test cần update:**
- `TOLERANCE` trong test: `0.5` → `0.60` (SFace cosine)
- `CONFUSION_MARGIN`: `0.10` → `0.12`

---

## 3.9 Tiêu Chí Kiểm Tra Phase 3

```bash
# 1. Models load được
python -c "
import cv2
d = cv2.FaceDetectorYN.create('src/yolo/yunet_int8.onnx', '', (320, 240))
r = cv2.FaceRecognizerSF.create('model/sface_int8.onnx', '')
print('Both models loaded OK')
"

# 2. Detect + encode 1 ảnh thật
python -c "
import cv2, numpy as np
from core.detector import YuNetDetector
from core.embedder import SFaceEmbedder
frame = cv2.imread('family_images/Khai/<any_image>.jpg')
det = YuNetDetector('src/yolo/yunet_int8.onnx')
emb = SFaceEmbedder('model/sface_int8.onnx')
detections = det.detect(frame)
print(f'Detected {len(detections)} faces')
if detections:
    enc = emb.encode(frame, detections[0])
    print(f'Encoding shape: {enc.shape}, norm: {np.linalg.norm(enc):.3f}')
"

# 3. Rebuild encodings hoàn tất
FACE_BACKEND=sface python -c "from core.encoder import rebuild_encodings; rebuild_encodings()"
# Xem output: mỗi người có đủ encodings sau clustering

# 4. Test suite pass
FACE_BACKEND=sface pytest tests/ -v

# 5. Live recognition test (với khuôn mặt thật)
FACE_BACKEND=sface HEADLESS=false python src/main.py
# Quan sát: nhận đúng tên người nhà, "Unknown" cho người lạ

# 6. Dependency drop check
python -c "import dlib"   # ModuleNotFoundError sau khi uninstall (cuối Phase 3)
```

---

## Rollback Phase 3 (Nếu Accuracy Kém)

```bash
# 1. Chuyển về dlib trong .env:
echo "FACE_BACKEND=dlib" >> .env

# 2. Restore backup encodings:
cp model/face_encodings_dlib_backup_YYYYMMDD.pkl model/face_encodings_hybrid.pkl

# 3. Reinstall nếu đã uninstall:
pip install dlib face-recognition ultralytics onnxruntime

# 4. Hệ thống chạy lại với stack cũ — Phase 1 vẫn còn hiệu lực (headless, throttle, v.v.)
```
