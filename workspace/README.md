# Family Face Recognition System

Hệ thống nhận diện khuôn mặt gia đình kết hợp YOLO (face detection) và dlib / face_recognition (face identification). Hỗ trợ thu thập dữ liệu từ video/camera, augment dataset, tối ưu encodings, và export ONNX để tăng tốc trên CPU.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11/v12-orange.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey.svg)

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Tối Ưu Hiệu Suất CPU](#tối-ưu-hiệu-suất-cpu)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Tham Số Hệ Thống](#tham-số-hệ-thống)
- [Tài Liệu Thiết Kế](#tài-liệu-thiết-kế)
- [Testing](#testing)
- [Xử Lý Sự Cố](#xử-lý-sự-cố)

---

## Tổng Quan

### Tính Năng Chính

| Tính năng | Mô tả |
|-----------|--------|
| **Nhận diện real-time** | Camera → YOLO detect → face_recognition identify → hiển thị tên |
| **Thu thập từ video** | Tự động chọn frame chất lượng, đa dạng từ video file |
| **Thu thập từ camera** | Ghi video → multi-pass extract frames |
| **Quản lý persons** | CRUD: tạo, xem, xóa, reset người trong dataset |
| **Augment dataset** | Tạo 8 biến thể ảnh (blur, dark, noise, ...) |
| **Tối ưu encodings** | Clustering giảm 90%+ encoding trùng lặp |
| **Export ONNX** | YOLO .pt → .onnx tăng tốc ~1.5-2x trên CPU |
| **Benchmark models** | So sánh hiệu năng các YOLO model |

### Quy Trình Hoạt Động

```
[1] Export ONNX (1 lần)
         ↓
[2] Thu thập dữ liệu → [3] Augment (tùy chọn) → [4] Rebuild encodings → [5] Optimize (tùy chọn)
         ↓
[6] Nhận diện real-time
```

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                                  │
│                                                                  │
│  src/face_manager.py          src/main.py                       │
│  (cập nhật data / train lại)  (nhận diện real-time)             │
└────────────────────┬───────────────────┬────────────────────────┘
                     │                   │
         ┌───────────▼──┐    ┌───────────▼──────────────┐
         │  src/training/│    │       src/services/       │
         │               │    │                           │
         │ PersonManager │    │ AlertManager (Telegram)   │
         │ DataCollection│    │ DeviceDispatcher (MQTT)   │
         │ ExtractionSet │    │ EventLogger (JSONL)       │
         │ InteractiveCLI│    │ NotificationWorker        │
         └───────────────┘    └───────────────────────────┘
                     │                   │
                     └─────────┬─────────┘
                               │
         ┌─────────────────────▼─────────────────────────┐
         │                  src/core/                     │
         │                                               │
         │  recognizer.py    — FaceRecognizer (camera)   │
         │  encoder.py       — rebuild_encodings()       │
         │  face_utils.py    — extract_face_region()     │
         │  frame_extractor.py — VideoFrameExtractor     │
         │  frame_quality.py   — FrameQualityChecker     │
         │  frame_diversity.py — FrameDiversityFilter    │
         │  motion_guard.py    — MotionGuard             │
         │  recognition_stabilizer.py — anti-noise       │
         └───────────────────────────────────────────────┘

scripts/ (chạy độc lập, maintenance tools)
  augment_dataset.py   optimize_encodings.py
  export_onnx.py       benchmark_models.py
```

### Pipeline nhận diện real-time (chi tiết)

```
Camera (1280×720)`
    │
    ▼ [Capture Thread]
Frame Queue (maxsize=1) — luôn lấy frame mới nhất
    │
    ▼
Resize 1280 → 416px ──────────────────── YOLO_INPUT_WIDTH=416
    │
    ▼
YOLO Detection (ONNX Runtime) ─────────── ~20-40ms/frame CPU
    │ scale bbox ×(1280/416)
    ▼
BGR → RGB (1 lần/frame)
    │
    ▼
IoU Cache Check ──────────────────────── RECOGNITION_INTERVAL=10
    ├─ Cache HIT (93% frames)  → reuse name
    └─ Cache MISS (7% frames)  → face_recognition.face_encodings()
                                    skip dlib HOG → dùng YOLO bbox
                                    → 128-d vector → face_distance()
    │
    ▼
draw_results() + imshow()

Hiệu suất (CPU, không GPU):
  Trước tối ưu: ~10-17 FPS
  Sau tối ưu:   ~22-30 FPS  (+50-80%)
```

Xem chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Cài Đặt

### Yêu Cầu

```bash
python --version   # 3.8+

# Core
pip install opencv-python numpy ultralytics face_recognition cmake dlib
pip install questionary scikit-image tqdm

# ONNX (tùy chọn nhưng khuyến nghị — tăng tốc CPU)
pip install onnxruntime

# Testing
pip install pytest pytest-mock
```

**Lưu ý cài đặt dlib:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev libboost-python-dev cmake
pip install dlib
# Hoặc nếu lỗi:
pip install dlib-bin
```

### Kiểm Tra Cài Đặt

```bash
cd /home/ubuntu/workspace
python -c "import cv2, face_recognition, ultralytics; print('OK')"
python -c "import onnxruntime; print('ONNX Runtime OK')"
```

---

## Hướng Dẫn Sử Dụng

### Bước 1 — Export ONNX (chạy 1 lần, khuyến nghị)

```bash
python scripts/export_onnx.py
```

Script sẽ:
- Export `yolov11n-face.pt` → `yolov11n-face.onnx`
- Tự benchmark so sánh PyTorch vs ONNX Runtime
- Sau đó `config.py` tự động dùng `.onnx`

```bash
# Tùy chọn: export với imgsz khác
python scripts/export_onnx.py --model src/yolo/yolov11n-face.pt --imgsz 416
```

### Bước 2 — Thu Thập Dữ Liệu (Menu Chính)

```bash
python src/face_manager.py
```

Menu cung cấp 7 chức năng:
1. **Thu thập dữ liệu mới** — từ camera hoặc video file
2. **Quản lý Persons** — xem, xóa, reset
3. **Augment dataset** — tạo biến thể ảnh
4. **Rebuild encodings** — tạo lại encodings từ dataset
5. **Optimize encodings** — clustering giảm trùng lặp
6. **Cài đặt thu thập frame** — tune tham số
7. **Trợ giúp** — hướng dẫn chi tiết

### Bước 3 — Nhận Diện Real-Time

```bash
python src/main.py

# Tùy chọn:
python src/main.py --camera 0 --tolerance 0.6 --detection-confidence 0.3
python src/main.py --recognition-interval 10 --yolo-input-width 416
```

**Điều khiển:** Nhấn `q` để thoát.

### Các Lệnh Khác

```bash
# Augment dataset
python scripts/augment_dataset.py                    # toàn bộ dataset
python scripts/augment_dataset.py --person Khai      # 1 người cụ thể
python scripts/augment_dataset.py --no-preview       # bỏ qua preview

# Tối ưu encodings
python scripts/optimize_encodings.py

# Benchmark YOLO models
python scripts/benchmark_models.py
python scripts/benchmark_models.py --frames 150 --sample 8
```

---

## Tối Ưu Hiệu Suất CPU

Hệ thống đã tích hợp 3 tối ưu cho máy không có GPU:

### 1. ONNX Runtime (khuyến nghị, +30-50% FPS)

```bash
pip install onnxruntime
python scripts/export_onnx.py          # Export 1 lần
# config.py tự động dùng .onnx từ đây
```

ONNX Runtime nhanh hơn PyTorch ~1.5-2x trên CPU vì:
- Static graph (không overhead từ dynamic graph PyTorch)
- CPU-specific kernel optimizations (SIMD, threading pool)

### 2. YOLO Input Width 416px (đã áp dụng)

`YOLO_INPUT_WIDTH = 416` trong `config.py` — tăng tốc YOLO ~1.5-2x so với 640px, giữ đủ độ phân giải cho khuôn mặt gần và trung bình.

> Nếu cần nhận diện khuôn mặt ở xa (nhỏ trên frame), đổi lại 640.

### 3. Recognition Interval 10 (đã áp dụng)

`RECOGNITION_INTERVAL = 10` trong `config.py` — re-encode mỗi 10 frames thay vì 5, giảm 50% số lần gọi dlib encoding. Lag nhận diện ~0.5s tại 20 FPS — không nhận thấy bằng mắt thường.

### Hiệu suất tổng hợp

| Cấu hình | FPS (CPU) |
|----------|-----------|
| PyTorch + 640px + interval=5 (cũ) | ~10-17 |
| ONNX + 416px + interval=10 (mới) | ~22-30 |

---

## Cấu Trúc Dự Án

```
workspace/
├── README.md
├── extraction_settings.json     # Tham số trích xuất frame (persistent)
│
├── docs/
│   ├── ARCHITECTURE.md          # Kiến trúc tổng thể (v4.0)
│   ├── DATA_FLOW.md             # Luồng dữ liệu chi tiết
│   ├── API.md                   # API specification
│   └── DESIGN.md                # Thiết kế hệ thống
│
├── src/
│   ├── face_manager.py          # ← chạy để cập nhật data / train lại
│   ├── main.py                  # ← chạy để nhận diện real-time
│   ├── config.py                # Tất cả tham số hệ thống (auto-detect ONNX/PT)
│   │
│   ├── core/                    # Reusable — không chứa business logic
│   │   ├── recognizer.py        # FaceRecognizer — camera loop + detection
│   │   ├── encoder.py           # rebuild_encodings(), update_person_encodings()
│   │   ├── face_utils.py        # extract_face_region()
│   │   ├── frame_extractor.py   # VideoFrameExtractor + ExtractedFrame
│   │   ├── frame_quality.py     # FrameQualityChecker (sharp/bright/size)
│   │   ├── frame_diversity.py   # FrameDiversityFilter (SSIM dedup)
│   │   ├── motion_guard.py      # MotionGuard — bỏ qua frame không có chuyển động
│   │   └── recognition_stabilizer.py  # Anti-noise cho nhận diện real-time
│   │
│   ├── services/                # Smart home business logic
│   │   ├── alert_manager.py     # Cảnh báo người lạ (Telegram, sound)
│   │   ├── device_dispatcher.py # Điều khiển thiết bị (MQTT, Serial)
│   │   ├── event_logger.py      # Ghi lịch sử vào nhà (JSONL)
│   │   └── notification_worker.py  # Background thread xử lý I/O
│   │
│   ├── training/                # Training domain
│   │   ├── cli.py               # InteractiveCLI — giao diện dòng lệnh
│   │   ├── person_manager.py    # PersonManager — CRUD persons
│   │   ├── data_collection.py   # DataCollectionSession — thu thập ảnh
│   │   └── extraction_settings.py  # ExtractionSettings + presets
│   │
│   └── yolo/
│       ├── yolov11n-face.pt     # Nano — default
│       ├── yolov11n-face.onnx   # ONNX version — nhanh hơn ~1.5-2x [sau export]
│       ├── yolov11s-face.pt     # Small — cân bằng
│       └── yolov12s-face.pt     # v12 Small — chính xác nhất
│
├── scripts/                     # Maintenance tools (chạy độc lập)
│   ├── augment_dataset.py       # Augment 8 biến thể ảnh
│   ├── optimize_encodings.py    # Clustering — giảm encoding trùng lặp
│   ├── export_onnx.py           # Export YOLO .pt → .onnx
│   └── benchmark_models.py      # Benchmark các YOLO model
│
├── model/
│   └── face_encodings_hybrid.pkl # Face encodings database
│
├── family_images/                # Training dataset
│   └── <Person>/                 # Ảnh gốc + augmented per person
│
├── data/                         # Raw video recordings
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_frame_quality.py    # Unit: FrameQualityChecker
│   ├── test_frame_extractor.py  # Unit: DiversityFilter, VideoFrameExtractor
│   ├── test_person_manager.py   # Unit: PersonManager CRUD
│   ├── test_alert_manager.py    # Unit: AlertManager
│   ├── test_device_dispatcher.py # Unit: DeviceDispatcher backends
│   ├── test_event_logger.py     # Unit: EventLogger
│   ├── test_stabilizer.py       # Unit: RecognitionStabilizer
│   └── test_integration.py      # Integration: full pipeline
│
└── benchmark_results/            # Benchmark output images
```

---

## Tham Số Hệ Thống

Tất cả tham số tập trung trong `src/config.py`:

### Hiệu suất real-time

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `YOLO_INPUT_WIDTH` | **416** | Resize frame trước YOLO — nhanh hơn 640 ~1.5-2x |
| `RECOGNITION_INTERVAL` | **10** | Re-encode mỗi N frames — cao hơn = nhanh hơn nhưng lag nhận diện |
| `MODEL_PATH` | auto | Dùng `.onnx` nếu có, fallback `.pt` |

### Nhận diện

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `TOLERANCE` | 0.6 | Ngưỡng nhận diện (thấp = strict hơn) |
| `DETECTION_CONFIDENCE` | 0.3 | YOLO detect nhạy hơn khi giảm |
| `RECOGNITION_PADDING` | 15px | Padding quanh bbox khi encode |

### Thu thập frame

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `FRAME_MIN_LAPLACIAN` | 15 | Frame nét hơn khi tăng |
| `FRAME_MIN_FACE_PX` | 70 | Chấp nhận mặt nhỏ hơn khi giảm |
| `FRAME_DETECT_CONF` | 0.15 | Detect nhiều mặt hơn khi giảm |

### Encoding

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `MAX_ENCODINGS_PER_PERSON` | 30 | Số encoding tối đa sau optimize |
| `CLUSTERING_THRESHOLD` | 0.15 | Gộp encoding giống nhau hơn khi tăng |

**Presets thu thập frame** (trong InteractiveCLI → Cài đặt):
- **Thoai mai:** Bắt nhiều frame, ít reject nhất
- **Binh thuong:** Mặc định, khuyến nghị
- **Khat khe:** Chỉ frame chất lượng cao

---

## Tài Liệu Thiết Kế

| Document | Nội dung |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc tổng thể v3.0, pipeline chi tiết, quyết định thiết kế |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | Luồng dữ liệu: thu thập, nhận diện, augment, optimize |
| [docs/API.md](docs/API.md) | API specification: classes, methods, parameters, return types |
| [docs/DESIGN.md](docs/DESIGN.md) | Thiết kế training manager: frame extraction, test plan |

---

## Testing

```bash
cd /home/ubuntu/workspace
python -m pytest tests/ -v

# Chạy cụ thể:
python -m pytest tests/test_frame_quality.py -v    # Unit: quality checker
python -m pytest tests/test_person_manager.py -v   # Unit: person CRUD
python -m pytest tests/test_frame_extractor.py -v  # Unit: diversity + extractor
python -m pytest tests/test_integration.py -v      # Integration: full pipeline
```

**40+ tests** covering:
- Frame quality checks (sharpness, brightness, size)
- Person CRUD operations
- Frame diversity filtering (SSIM)
- Video frame extraction
- End-to-end data collection

---

## Xử Lý Sự Cố

| Vấn đề | Giải pháp |
|--------|-----------|
| Không mở được camera | `ls /dev/video*`, kiểm tra quyền |
| YOLO model không tìm thấy | Kiểm tra `src/yolo/*.pt` |
| face_recognition import error | `pip install cmake dlib face_recognition` |
| onnxruntime không tìm thấy | `pip install onnxruntime` |
| ONNX chưa có, dùng PyTorch | Chạy `python scripts/export_onnx.py` |
| Nhận diện không chính xác | Thêm ảnh training, giảm `TOLERANCE` |
| Nhận diện chậm (CPU) | Chạy export_onnx.py, YOLO_INPUT_WIDTH=416 |
| FPS thấp dù đã ONNX | Tăng `RECOGNITION_INTERVAL` lên 15 |
| Frame extract quá ít | Giảm `min_laplacian`, `min_face_px` trong settings |
| Bỏ sót mặt ở xa | Tăng `YOLO_INPUT_WIDTH` lên 640 |
