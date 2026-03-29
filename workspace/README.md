# Family Face Recognition System

Hệ thống nhận diện khuôn mặt gia đình kết hợp YOLO (face detection) và dlib / face_recognition (face identification). Hỗ trợ thu thập dữ liệu từ video/camera, augment dataset, và tối ưu encodings để nhận diện real-time.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11/v12-orange.svg)

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Tài Liệu Thiết Kế](#tài-liệu-thiết-kế)
- [Tham Số Hệ Thống](#tham-số-hệ-thống)
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
| **Benchmark models** | So sánh hiệu năng các YOLO model |

### Quy Trình Hoạt Động

```
Thu thập dữ liệu → Augment (tùy chọn) → Rebuild encodings → Optimize (tùy chọn) → Nhận diện
```

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE CLI                               │
│               training_manager.py                                │
│                                                                  │
│  ┌──────────────┐ ┌────────────────────┐ ┌──────────────────┐   │
│  │PersonManager  │ │DataCollectionSession│ │ExtractionSettings│   │
│  │ CRUD persons  │ │ run_from_file()    │ │ Presets / tuning │   │
│  │              │ │ run_from_camera()  │ │                  │   │
│  └──────────────┘ └────────┬───────────┘ └──────────────────┘   │
│                            │                                     │
│  ┌──────────────┐ ┌───────▼─────────────────────────────────┐   │
│  │augment_      │ │       VideoFrameExtractor               │   │
│  │dataset.py    │ │                                         │   │
│  │ 8 biến thể   │ │  ┌─────────────────┐ ┌───────────────┐  │   │
│  └──────────────┘ │  │QualityChecker    │ │DiversityFilter│  │   │
│                    │  │ sharp/bright/size│ │ SSIM dedup    │  │   │
│  ┌──────────────┐ │  └─────────────────┘ └───────────────┘  │   │
│  │optimize_     │ └────────────────────────────────────────┘   │
│  │encodings.py  │                                               │
│  │ clustering   │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
┌──────────────┐   ┌────────────────────────┐
│  RECOGNIZER  │   │      DATA STORES       │
│ recognizer.py│   │                        │
│              │   │ family_images/<Person>/ │
│ YOLO detect  │   │ model/*.pkl            │
│ → encode     │   │ data/*.mp4             │
│ → match DB   │   │ extraction_settings.json│
│ → display    │   └────────────────────────┘
└──────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│            CORE UTILITIES                │
│  core/face_utils.py    — crop face       │
│  core/encoder.py       — rebuild pkl     │
│  core/frame_extractor.py — video extract │
└──────────────────────────────────────────┘
```

Xem chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Cài Đặt

### Yêu Cầu

```bash
python --version   # 3.8+
pip install opencv-python numpy ultralytics face_recognition cmake dlib
pip install questionary scikit-image tqdm
pip install pytest pytest-mock  # for testing
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
cd /home/dkhai/workspace
python -c "import cv2, face_recognition, ultralytics; print('OK')"
```

---

## Hướng Dẫn Sử Dụng

### Menu Chính (Khuyến nghị)

```bash
cd /home/dkhai/workspace/src
python training_manager.py
```

Menu cung cấp 7 chức năng:
1. **Thu thập dữ liệu mới** — từ camera hoặc video file
2. **Quản lý Persons** — xem, xóa, reset
3. **Augment dataset** — tạo biến thể ảnh
4. **Rebuild encodings** — tạo lại encodings từ dataset
5. **Optimize encodings** — clustering giảm trùng lặp
6. **Cài đặt thu thập frame** — tune tham số
7. **Trợ giúp** — hướng dẫn chi tiết

### Nhận Diện Real-Time

```bash
cd /home/dkhai/workspace/src
python recognizer.py

# Tùy chọn:
python recognizer.py --camera 0 --tolerance 0.6 --detection-confidence 0.3
python recognizer.py --model yolo/yolov11n-face.pt  # model nhanh hơn
```

**Điều khiển:** Nhấn `q` để thoát.

### Augment Dataset

```bash
cd /home/dkhai/workspace/src
python augment_dataset.py                    # toàn bộ dataset
python augment_dataset.py --person Khai      # 1 người cụ thể
python augment_dataset.py --no-preview       # bỏ qua preview
```

### Tối Ưu Encodings

```bash
cd /home/dkhai/workspace/src
python optimize_encodings.py
```

### Benchmark Models

```bash
cd /home/dkhai/workspace/src
python benchmark_models.py
python benchmark_models.py --frames 150 --sample 8
```

---

## Cấu Trúc Dự Án

```
/home/dkhai/workspace/
├── README.md                    # File này
├── extraction_settings.json     # Tham số trích xuất frame (persistent)
│
├── docs/
│   ├── DESIGN.md                # Thiết kế training manager
│   ├── ARCHITECTURE.md          # Kiến trúc tổng thể (MỚI)
│   ├── DATA_FLOW.md             # Luồng dữ liệu chi tiết (MỚI)
│   └── API.md                   # API specification (MỚI)
│
├── src/
│   ├── config.py                # Centralized configuration
│   ├── recognizer.py            # Nhận diện real-time (FaceRecognizer)
│   ├── training_manager.py      # CLI quản lý + thu thập (InteractiveCLI)
│   ├── augment_dataset.py       # Augment 8 biến thể
│   ├── optimize_encodings.py    # Clustering encodings
│   ├── benchmark_models.py      # Benchmark YOLO models
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── face_utils.py        # extract_face_region()
│   │   ├── encoder.py           # rebuild_encodings()
│   │   └── frame_extractor.py   # VideoFrameExtractor, QualityChecker, DiversityFilter
│   │
│   └── yolo/
│       ├── yolov11n-face.pt     # Nano — nhanh nhất (~17 FPS)
│       ├── yolov11s-face.pt     # Small — cân bằng
│       └── yolov12s-face.pt     # v12 Small — chính xác nhất (~7 FPS)
│
├── model/
│   └── face_encodings_hybrid.pkl # Face encodings database
│
├── family_images/                # Training dataset
│   ├── Khai/                    # 50+ ảnh
│   └── MaiAnh/                  # 50+ ảnh
│
├── data/                         # Raw video recordings
│   ├── Khai_*.mp4
│   └── MaiAnh_*.mp4
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_frame_quality.py    # Unit: FrameQualityChecker
│   ├── test_person_manager.py   # Unit: PersonManager CRUD
│   ├── test_frame_extractor.py  # Unit: DiversityFilter, Extractor
│   └── test_integration.py      # Integration: full pipeline
│
└── benchmark_results/            # Benchmark output images
```

---

## Tài Liệu Thiết Kế

| Document | Nội dung |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc tổng thể, các thành phần, pipeline chính, quyết định thiết kế |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | Luồng dữ liệu chi tiết: thu thập, nhận diện, augment, optimize |
| [docs/API.md](docs/API.md) | API specification: classes, methods, parameters, return types |
| [docs/DESIGN.md](docs/DESIGN.md) | Thiết kế training manager: frame extraction pipeline, test plan |

---

## Tham Số Hệ Thống

Tất cả tham số tập trung trong `src/config.py`:

| Tham số | Default | Ảnh hưởng |
|---------|---------|-----------|
| `TOLERANCE` | 0.6 | Ngưỡng nhận diện (thấp = strict hơn) |
| `DETECTION_CONFIDENCE` | 0.3 | YOLO detect nhạy hơn khi giảm |
| `FRAME_MIN_LAPLACIAN` | 15 | Frame nét hơn khi tăng |
| `FRAME_MIN_FACE_PX` | 70 | Chấp nhận mặt nhỏ hơn khi giảm |
| `FRAME_DETECT_CONF` | 0.15 | Detect nhiều mặt hơn khi giảm |
| `MAX_ENCODINGS_PER_PERSON` | 30 | Số encoding tối đa sau optimize |
| `CLUSTERING_THRESHOLD` | 0.15 | Gộp encoding giống nhau hơn khi tăng |

**Presets thu thập frame** (trong InteractiveCLI → Cài đặt):
- **Thoai mai:** Bắt nhiều frame, ít reject nhất
- **Binh thuong:** Mặc định, khuyến nghị
- **Khat khe:** Chỉ frame chất lượng cao

---

## Testing

```bash
cd /home/dkhai/workspace
pytest tests/ -v

# Chạy cụ thể:
pytest tests/test_frame_quality.py -v    # Unit: quality checker
pytest tests/test_person_manager.py -v   # Unit: person CRUD
pytest tests/test_frame_extractor.py -v  # Unit: diversity + extractor
pytest tests/test_integration.py -v      # Integration: full pipeline
```

**37 tests** covering:
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
| Nhận diện không chính xác | Thêm ảnh training, giảm `TOLERANCE` |
| Nhận diện chậm | Chạy `optimize_encodings.py`, dùng `yolov11n` |
| Frame extract quá ít | Giảm `min_laplacian`, `min_face_px` trong settings |
