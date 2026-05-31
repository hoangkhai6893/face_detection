# Kiến Trúc Nâng Cấp — Family Face Recognition System

> **Phiên bản:** 1.0 · **Ngày:** 2026-06-01
> **Dev env:** Intel NUC (x86) · **Deploy target:** Raspberry Pi 5 (4GB), headless, 24/7

---

## 1. Tóm Tắt Điều Hướng

| | Stack Hiện Tại | Stack Mục Tiêu |
|---|---|---|
| **Detector** | YOLOv11n (ONNX/PT, ultralytics) | YuNet (`cv2.FaceDetectorYN`, INT8 ONNX) |
| **Embedder** | dlib ResNet 128-d (`face_recognition`) | SFace 128-d (`cv2.FaceRecognizerSF`, INT8 ONNX) |
| **Distance** | Euclidean (lower = closer) | Cosine via `1 - score` (lower = closer) |
| **Display** | `cv2.imshow` vô điều kiện | Guarded `if not headless` |
| **ACTIVE CPU** | 100% mỗi frame | Throttle mỗi N frame |
| **Dependencies** | dlib + face_recognition + ultralytics + onnxruntime | **chỉ OpenCV + 2 ONNX files** |
| **Deploy Pi** | Không chạy được (imshow crash) + build dlib khó | Chạy ngay, deploy đơn giản |

**Mục tiêu lâu dài:** mỗi Pi 5 độc lập chạy 1 camera, tiêu thụ ~2% CPU khi nhà trống, ~40-50% khi có người, tự khởi động lại khi crash.

---

## 2. Kiến Trúc Hiện Tại

```
┌─────────────────────────────────────────────────────────┐
│                    src/main.py                          │
│  CLI args → FaceRecognizer + AlertMgr + Dispatcher      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               src/core/recognizer.py                    │
│                                                         │
│  Camera (1280×720) ──→ MotionGuard.should_process()     │
│                              │                          │
│                         IDLE (~2% CPU)                  │
│                         absdiff + MOG2 on 160×90        │
│                              │ motion / probe timeout   │
│                         ACTIVE (full pipeline)          │
│                              │                          │
│  Resize to 416px ──→ YOLO.detect() [ultralytics]        │
│  ~30-60ms/frame              │                          │
│                         Per face (IOU cache miss):      │
│  face_recognition.face_encodings()  [dlib ResNet]       │
│  ~30-80ms/face               │                          │
│  face_recognition.face_distance()  [numpy, ~0.5ms]      │
│  Top-K weighted vote → winner + margin check            │
│                              │                          │
│  cv2.imshow() [TỰ ĐỘNG, KHÔNG CÓ GUARD]                 │ ← crash Pi headless
│  cv2.waitKey()                                          │
└────────────────────┬────────────────────────────────────┘
                     │ on_recognition_update callback
┌────────────────────▼────────────────────────────────────┐
│         RecognitionStabilizer (voting 2/5 & 4/5)        │
└────────────────────┬────────────────────────────────────┘
                     │ StableEvent (non-blocking queue)
┌────────────────────▼────────────────────────────────────┐
│               NotificationWorker (daemon thread)        │
│  EventLogger │ AlertManager (Telegram) │ DeviceDispatcher│
└─────────────────────────────────────────────────────────┘
```

**Điểm nghẽn chính:**
- YOLO + dlib chạy mọi frame ACTIVE → ~70-80% CPU Pi5
- `cv2.imshow` unconditional → crash Pi headless
- `face_recognition` (dlib) khó build trên ARM → deploy phức tạp

---

## 3. Kiến Trúc Mục Tiêu (Sau Phase 3)

```
┌─────────────────────────────────────────────────────────┐
│                    src/main.py                          │
│  --headless flag + HEADLESS env var                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               src/core/recognizer.py                    │
│                                                         │
│  Camera (640×480) ──→ MotionGuard.should_process()      │
│                              │                          │
│                         IDLE (~2% CPU)  ← GIỮ NGUYÊN   │
│                              │                          │
│                         ACTIVE (throttle mỗi N frame)  │
│                              │ (ACTIVE_PROCESS_EVERY=2) │
│                              │                          │
│  FaceDetector.detect(frame)  ← interface                │
│  [YuNetDetector, ~5-15ms]    │                          │
│                         Per face (IOU cache miss):      │
│  FaceEmbedder.encode(frame, detection) ← interface      │
│  [SFaceEmbedder: alignCrop + feature, ~20-50ms/face]    │
│  FaceEmbedder.distance(a, b) → cosine distance          │
│  Top-K weighted vote → winner + margin check            │
│                              │                          │
│  if not headless: draw_results() + cv2.imshow           │ ← guarded
└────────────────────┬────────────────────────────────────┘
                     │ callback (giữ nguyên)
┌────────────────────▼────────────────────────────────────┐
│  RecognitionStabilizer  (GIỮ NGUYÊN — model-agnostic)   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  NotificationWorker (GIỮ NGUYÊN — model-agnostic)       │
│  EventLogger (rotation) │ AlertManager │ Dispatcher      │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Cấu Trúc File — Thay Đổi Qua Các Phase

```
src/
  main.py                        [P1] --headless flag
  config.py                      [P1+P3] headless, throttle, camera, cosine thresholds
  core/
    recognizer.py                [P1+P2+P3] headless guard, throttle, interface calls
    detector.py                  [P2] NEW — FaceDetector ABC + YoloDetector + YuNetDetector
    embedder.py                  [P2] NEW — FaceEmbedder ABC + DlibEmbedder + SFaceEmbedder
    encoder.py                   [P3] SFace-aware rebuild_encodings()
    frame_extractor.py           [P3] YuNet-based detection + fix detect_conf bug
    motion_guard.py              ← KHÔNG ĐỔI (probe burst đã tốt)
    recognition_stabilizer.py   ← KHÔNG ĐỔI (model-agnostic)
    frame_quality.py             ← KHÔNG ĐỔI
    frame_diversity.py           ← KHÔNG ĐỔI
  services/
    event_logger.py              [P1] log rotation theo ngày
    notification_worker.py       ← KHÔNG ĐỔI
    alert_manager.py             ← KHÔNG ĐỔI
    device_dispatcher.py         ← KHÔNG ĐỔI
scripts/
  install_service.sh             [P1] NEW — systemd Pi5
  validate_backends.py           [P2] NEW — A/B accuracy+speed harness
model/
  face_detection_yunet_*_int8.onnx   [P3] download từ OpenCV Zoo
  face_recognition_sface_*_int8.onnx [P3] download từ OpenCV Zoo
  face_encodings_hybrid.pkl          ← rebuild sau P3
```

---

## 5. Migration Roadmap Tổng Quan

```
Phase 1 ─────────────────────────────────────────────────
  Mục tiêu: Pi5 headless + 24/7 chạy được ngay với stack cũ
  Rủi ro: THẤP (không đổi model, không đổi accuracy)
  Rollback: git revert

Phase 2 ─────────────────────────────────────────────────
  Mục tiêu: abstraction layer + validation harness (gate)
  Rủi ro: THẤP (thêm code, không xóa code cũ)
  Gate: chỉ qua Phase 3 nếu accuracy(YuNet+SFace) >= accuracy(YOLO+dlib)

Phase 3 ─────────────────────────────────────────────────
  Mục tiêu: switch sang YuNet+SFace, bỏ dlib
  Rủi ro: TRUNG BÌNH (rebuild encodings, retune thresholds)
  Rollback: git revert + dùng lại .pkl backup

Phase 4 ─────────────────────────────────────────────────
  Mục tiêu: benchmark Pi5 thật + tune accuracy parameters
  Rủi ro: KHÔNG (chỉ đo + chỉnh config)
```

---

## 6. Interface Mới — FaceDetector & FaceEmbedder (Preview Phase 2)

Hai abstract classes đóng vai trò **firewall giữa stack model và business logic**. recognizer.py, encoder.py, frame_extractor.py chỉ phụ thuộc vào interface — không phụ thuộc ultralytics hay dlib.

```python
# src/core/detector.py
class FaceDetection(NamedTuple):
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original frame space
    confidence: float
    landmarks: Optional[np.ndarray]   # (5, 2) array hoặc None

class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]: ...
    def close(self) -> None: pass

# src/core/embedder.py
class FaceEmbedder(ABC):
    # distance semantics: LUÔN là 0 = same, 1 = very different
    # DlibEmbedder:  Euclidean (face_recognition.face_distance)
    # SFaceEmbedder: 1 - cosine_score (cv2.FaceRecognizerSF.FR_COSINE)

    @abstractmethod
    def encode(self, frame_bgr: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]: ...

    @abstractmethod
    def batch_distance(self, known: List[np.ndarray], query: np.ndarray) -> np.ndarray: ...
    # returns array of distances (same semantics for both backends)

    def close(self) -> None: pass
```

recognizer.py thay `self.yolo_model` + `face_recognition.*` bằng `self.detector` + `self.embedder`:
```python
detections = self.detector.detect(small_frame)  # → List[FaceDetection]
enc = self.embedder.encode(rgb_frame, detection) # → np.ndarray
dists = self.embedder.batch_distance(self.known_face_encodings, enc)
```

---

## 7. Thay Đổi Distance Space (QUAN TRỌNG cho Phase 3)

| | DlibEmbedder | SFaceEmbedder |
|---|---|---|
| Raw metric | L2 Euclidean | Cosine similarity (1=same, 0=diff) |
| Normalized distance | euclidean (0=same, 1+=diff) | `1 - cosine_score` (0=same, 1=diff) |
| Same person range | 0.0 – 0.50 | 0.01 – 0.637 |
| Different person range | 0.50 – 1.0+ | 0.637 – 1.0 |
| **TOLERANCE** | 0.50 | **~0.63** (cần đo lại với data thật) |
| **CONFUSION_MARGIN** | 0.10 | **~0.12 – 0.18** (cần đo) |

**Business logic (voting, margin check) trong recognizer.py KHÔNG đổi** — chỉ thay đổi giá trị config.

---

## 8. Model Files (Phase 3 Setup)

```bash
# Download từ OpenCV Zoo
OPENCV_ZOO=https://github.com/opencv/opencv_zoo/raw/main/models

wget $OPENCV_ZOO/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx -O src/yolo/yunet_int8.onnx
wget $OPENCV_ZOO/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx -O model/sface_int8.onnx
```

Config paths sau Phase 3:
```python
YUNET_MODEL_PATH = os.path.join(_SRC_DIR, "yolo", "yunet_int8.onnx")
SFACE_MODEL_PATH = os.path.join(_WORKSPACE_DIR, "model", "sface_int8.onnx")
```

---

## 9. Component Nào Được Tái Sử Dụng Nguyên Vẹn

| Component | Lý do giữ nguyên |
|---|---|
| `MotionGuard` (probe burst) | Model-agnostic — xử lý frame, không biết gì về model |
| `RecognitionStabilizer` | Chỉ nhận tên string + bbox — hoàn toàn model-agnostic |
| `NotificationWorker` | Queue pattern, không biết gì về recognition stack |
| `AlertManager` | Nhận frame + bbox, không biết detection stack |
| `DeviceDispatcher` | Nhận person name — model-agnostic |
| `EventLogger` | Ghi JSON — model-agnostic |
| `FrameQualityChecker` | OpenCV-only, không dùng YOLO/dlib |
| `FrameDiversityFilter` | OpenCV-only |

**Lợi ích kiến trúc:** services layer và core state machine hoàn toàn không cần đổi khi migrate model.
