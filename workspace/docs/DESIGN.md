# Thiết Kế Hệ Thống Thu Thập & Quản Lý Dữ Liệu Training

**Phiên bản:** 1.0
**Ngày:** 2026-03-28
**Trạng thái:** Approved — sẵn sàng implement

---

## 1. Bối Cảnh & Vấn Đề

### Hệ thống hiện tại
Dự án nhận diện khuôn mặt sử dụng:
- **YOLO v11** để detect khuôn mặt trong frame
- **dlib / face_recognition** để tạo 128-d face encoding và matching
- **Pickle file** `face_encodings_hybrid.pkl` lưu trữ tất cả encodings

### Vấn đề cần giải quyết

| Vấn đề | Mô tả |
|--------|-------|
| Quy trình phân tán | 4 script riêng biệt, không có flow thống nhất |
| Input kém hiệu quả | Chụp từng ảnh thủ công (SPACE), mất thời gian |
| Không có CRUD | Không thể xóa/reset person từ một nơi |
| Không có test | Mọi thứ kiểm tra thủ công |
| Thiếu frame diversity | Không có cơ chế tránh frame trùng lặp |

### Giải pháp
Tạo **1 chương trình thống nhất** (`training_manager.py`) với input từ **video** (file hoặc camera), tự động chọn frame đa dạng, chất lượng tốt, kết hợp interactive CLI để quản lý persons.

---

## 2. Kiến Trúc Tổng Quan

```
┌──────────────────────────────────────────────────────────────────┐
│                      training_manager.py                          │
│                      (Entry Point)                                 │
│                                                                    │
│  ┌──────────────────┐     ┌────────────────────────────────────┐  │
│  │  InteractiveCLI   │     │       DataCollectionSession        │  │
│  │                   │────▶│  Orchestrates data collection      │  │
│  │  - main_menu()    │     │  - run_from_file()                 │  │
│  │  - person select  │     │  - run_from_camera()               │  │
│  │  - source select  │     │  - _save_frame()                   │  │
│  └──────────────────┘     └──────────────┬─────────────────────┘  │
│                                          │                          │
│  ┌──────────────────┐     ┌─────────────▼───────────────────────┐ │
│  │  PersonManager    │     │        VideoFrameExtractor           │ │
│  │                   │     │  (core/frame_extractor.py)          │ │
│  │  - list_persons() │     │                                      │ │
│  │  - create_person()│     │  ┌──────────────────────────────┐   │ │
│  │  - delete_person()│     │  │    FrameQualityChecker        │   │ │
│  │  - reset_person() │     │  │  - is_sharp()                 │   │ │
│  └──────────────────┘      │  │  - is_bright()                │   │ │
│                            │  │  - is_large_enough()          │   │ │
│                            │  │  - score()                    │   │ │
│                            │  └──────────────────────────────┘   │ │
│                            │  ┌──────────────────────────────┐   │ │
│                            │  │    FrameDiversityFilter       │   │ │
│                            │  │  - is_diverse() (SSIM)        │   │ │
│                            │  │  - accept()                   │   │ │
│                            │  └──────────────────────────────┘   │ │
│                            └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  family_images/<person>/        Core utilities (REUSED):
  ├── <name>_<ts>_q<s>.jpg       - core/face_utils.py
  └── ...                        - core/encoder.py
                                  - config.py
```

---

## 3. Luồng Hoạt Động

### 3.1 Menu Chính

```
[Main Menu]
  ├── 1. Thu thập dữ liệu mới
  │    ├── Chọn / Tạo person  ──────────────────────────────┐
  │    │    ├── [Danh sách persons hiện có] → chọn → UPDATE  │
  │    │    └── [+ Tạo mới] → nhập tên → CREATE             │
  │    │                                                      │
  │    ├── Chọn nguồn input                                   │
  │    │    ├── Camera (chọn camera_id)                       │
  │    │    └── Video file (nhập đường dẫn)                  │
  │    │                                                      │
  │    └── Chạy collection → Hiển thị progress → Summary     │
  │         └── Hỏi: Augment? → Hỏi: Optimize?              │
  │                                                           │
  ├── 2. Quản lý Persons                                      │
  │    ├── Xem danh sách (tên, số ảnh, ngày tạo)             │
  │    ├── Xóa person → xác nhận → xóa thư mục → rebuild     │
  │    └── Reset person → xác nhận → xóa ảnh → rebuild       │
  │                                                           │
  ├── 3. Augment dataset                                      │
  ├── 4. Optimize encodings                                   │
  └── 5. Thoát
```

### 3.2 Video Frame Selection Pipeline

```
Video file / Camera stream
        │
        ▼  (skip mỗi FRAME_SKIP=5 frame)
┌───────────────────┐
│   YOLO Detection   │ ──── Không detect được ──▶ SKIP
└────────┬──────────┘
         │ Detect được face
         ▼
┌───────────────────┐
│ extract_face_      │  (reuse core/face_utils.py)
│ region() + padding │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│FrameQualityChecker│
│  is_sharp()?       │ ──── Laplacian var < 100 ──▶ SKIP (blurry)
│  is_bright()?      │ ──── brightness < 0.10   ──▶ SKIP (too dark)
│  is_large_enough()?│ ──── face < 100px        ──▶ SKIP (too small)
└────────┬──────────┘
         │ Tất cả pass
         ▼
┌───────────────────┐
│FrameDiversityFilter│
│  SSIM < 0.85?      │ ──── SSIM ≥ 0.85        ──▶ SKIP (duplicate)
│  frame_gap > 15?   │ ──── gap < 15 frames    ──▶ SKIP (too soon)
└────────┬──────────┘
         │ Diverse enough
         ▼
   SAVE face crop
   → family_images/<person>/<name>_<ts>_q<score>.jpg
```

---

## 4. API Specification

### 4.1 `core/frame_extractor.py`

#### `FrameQualityChecker`

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `is_sharp(frame, min_laplacian=100.0)` | BGR frame | `bool` | Laplacian var > threshold |
| `is_bright(frame, min=0.10, max=0.92)` | BGR frame | `bool` | Brightness trong khoảng chấp nhận được |
| `is_large_enough(face_crop, min_size=100)` | BGR crop | `bool` | min(h,w) >= min_size |
| `score(frame, face_crop)` | BGR frame + crop | `float [0,1]` | Weighted score: sharp 40%, bright 30%, size 30% |

#### `FrameDiversityFilter`

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `__init__(ssim_threshold=0.85, min_frame_gap=15)` | — | — | Khởi tạo với thresholds |
| `is_diverse(new_frame, frame_num)` | BGR frame, int | `bool` | SSIM < threshold AND gap đủ lớn |
| `accept(frame, frame_num)` | BGR frame, int | — | Cập nhật last accepted |
| `reset()` | — | — | Reset state (cho session mới) |

#### `VideoFrameExtractor`

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `extract_from_file(video_path, max_frames=200)` | str, int | `Iterator[ExtractedFrame]` | Đọc video file, yield frame đủ điều kiện |
| `extract_from_camera(camera_id=0, target_frames=50)` | int, int | `Iterator[ExtractedFrame]` | Capture từ camera, yield frame tốt |

#### `ExtractedFrame` (dataclass)

```python
@dataclass
class ExtractedFrame:
    face_crop: np.ndarray    # Face crop (với padding)
    quality_score: float      # Score 0-1
    source_frame_num: int     # Frame index trong video
    timestamp: float          # Giây tính từ đầu video
    detection_conf: float     # YOLO conf (0.0 = HOG fallback)
```

---

### 4.2 `training_manager.py`

#### `PersonManager`

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `list_persons()` | — | `List[PersonInfo]` | Danh sách persons với stats |
| `create_person(name)` | `str` | `Path` | Tạo thư mục mới |
| `delete_person(name, rebuild=True)` | `str, bool` | `bool` | Xóa thư mục + rebuild encodings |
| `reset_person(name, rebuild=True)` | `str, bool` | `int` | Xóa ảnh (giữ folder), trả về số ảnh đã xóa |
| `get_image_count(name)` | `str` | `int` | Đếm ảnh gốc (không tính `_aug_`) |

#### `PersonInfo` (dataclass)

```python
@dataclass
class PersonInfo:
    name: str
    image_count: int       # Số ảnh gốc
    augmented_count: int   # Số ảnh augmented
    folder_path: Path
    created_date: str      # ISO format
```

#### `DataCollectionSession`

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `run_from_file(video_path, max_frames=200)` | `str, int` | `CollectionResult` | Thu thập từ video file |
| `run_from_camera(camera_id=0, target_frames=50)` | `int, int` | `CollectionResult` | Thu thập từ camera |

#### `CollectionResult` (dataclass)

```python
@dataclass
class CollectionResult:
    person_name: str
    frames_processed: int
    frames_saved: int
    frames_rejected: int
    reject_reasons: Dict[str, int]  # {"blurry": 12, "dark": 5, "duplicate": 30, "no_face": 45}
    duration_seconds: float
    avg_quality: float
```

---

## 5. Cấu Trúc Files

```
/home/dkhai/workspace/
├── docs/
│   └── DESIGN.md                    ← File này
├── src/
│   ├── config.py                    (không đổi)
│   ├── training_manager.py          ← MỚI: Entry point chính
│   ├── core/
│   │   ├── __init__.py              (không đổi)
│   │   ├── face_utils.py            (không đổi — tái sử dụng)
│   │   ├── encoder.py               (không đổi — tái sử dụng)
│   │   └── frame_extractor.py       ← MỚI: Video frame selection
│   ├── improved_hybrid_recognition.py (không đổi)
│   ├── augment_dataset.py           (không đổi — được gọi từ menu)
│   └── optimize_encodings.py        (không đổi — được gọi từ menu)
└── tests/
    ├── __init__.py                  ← MỚI
    ├── conftest.py                  ← MỚI: Shared fixtures
    ├── test_frame_quality.py        ← MỚI: Unit tests quality checker
    ├── test_person_manager.py       ← MỚI: Unit tests CRUD
    ├── test_frame_extractor.py      ← MỚI: Unit tests extractor
    └── test_integration.py          ← MỚI: End-to-end tests
```

---

## 6. Conventions & Design Rules

### Filename Convention (không đổi)
```
family_images/<PersonName>/
├── <Name>_<timestamp_ms>_q<score>.jpg    ← từ camera/video (mới)
├── <Name>_<timestamp_ms>_conf<x.xx>.jpg  ← từ add_training_data (cũ)
└── <stem>_aug_<type>.jpg                 ← từ augment_dataset (cũ)
```

### Quy tắc thiết kế
1. **Config-driven**: Mọi threshold lấy từ `config.py`, không hardcode
2. **Tái sử dụng core/**: `extract_face_region()` và `rebuild_encodings()` không duplicate
3. **Không phá vỡ existing scripts**: Các script cũ vẫn chạy độc lập được
4. **Graceful degradation**: YOLO fail → HOG fallback; camera fail → thông báo rõ ràng
5. **Test isolation**: Test dùng `tmp_path` của pytest, không đụng dataset thật
6. **Idempotent**: Chạy lại không tạo file trùng (dùng timestamp ms)

---

## 7. Dependencies

| Package | Version | Mục đích |
|---------|---------|---------|
| `opencv-python` | hiện có | Video capture, image ops |
| `face_recognition` | hiện có | Face encoding |
| `ultralytics` | hiện có | YOLO detection |
| `numpy` | hiện có | Array ops |
| `questionary` | 2.1.1+ | Interactive CLI menu |
| `scikit-image` | 0.26.0+ | SSIM calculation |
| `tqdm` | 4.67.3+ | Progress bar cho video processing |
| `pytest` | 9.0.2+ | Test runner |
| `pytest-mock` | 3.15.1+ | Mock objects trong tests |

---

## 8. Test Plan

### Unit Tests

| Test file | Covers | Test count |
|-----------|--------|-----------|
| `test_frame_quality.py` | `FrameQualityChecker` — sharp/bright/size/score | 9 tests |
| `test_person_manager.py` | `PersonManager` — CRUD | 11 tests |
| `test_frame_extractor.py` | `FrameDiversityFilter`, `VideoFrameExtractor` | 9 tests |

### Integration Tests

| Test | Mô tả |
|------|-------|
| `test_full_video_collection` | Chạy collection từ synthetic video → kiểm tra ảnh được lưu |
| `test_person_create_then_collect` | Tạo person mới → thu thập → verify |
| `test_reset_person_clears_images` | Reset → verify folder trống |
| `test_collection_result_metrics` | Verify CollectionResult fields hợp lệ |

### Manual Tests (sau khi pytest xanh)

```bash
# Test interactive menu
python src/training_manager.py

# Test với video file thực
python src/training_manager.py
# → Thu thập → chọn person → Video file → đường dẫn video

# Test camera
# → Thu thập → Camera → Camera 0

# Verify recognition sau khi thu thập
python src/improved_hybrid_recognition.py
```

---

## 9. Lý Do Các Quyết Định Thiết Kế

### Tại sao `questionary` thay vì argparse?
- Argparse: người dùng phải nhớ flags, không hỏi về persons sẵn có
- Questionary: interactive, hiển thị danh sách persons, validate input ngay

### Tại sao SSIM thay vì pixel difference?
- Pixel diff: nhạy cảm với noise, shift nhỏ → báo "khác" dù thực ra giống
- SSIM: đánh giá cấu trúc tổng thể, bền hơn với noise và shift nhỏ

### Tại sao FRAME_SKIP=5?
- Video 30fps: skip 5 = xét mỗi 0.17s → đủ đa dạng về thời gian
- Video 60fps: skip 5 = xét mỗi 0.08s → diversity filter sẽ lọc thêm

### Tại sao max 200 frames từ video?
- 200 frames × 9 augmented = 1800 ảnh cho 1 người → quá đủ
- Tránh tốn disk space và thời gian optimize

### Tại sao tái sử dụng augment_dataset.py và optimize_encodings.py?
- Đã được test kỹ lưỡng, đã có fallback logic tốt
- Tránh duplicate code, giữ single source of truth
