# API Specification — Family Face Recognition System

**Phiên bản:** 2.1
**Ngày:** 2026-05-31

---

## 1. Module: `src/config.py`

Centralized configuration — tất cả paths và thresholds.

### Constants

| Tên | Type | Default | Mô tả |
|-----|------|---------|-------|
| `DATASET_PATH` | `str` | `<workspace>/family_images/` | Thư mục dataset |
| `MODEL_PATH` | `str` | `<src>/yolo/yolov12s-face.pt` | YOLO model file |
| `ENCODINGS_DIR` | `str` | `<workspace>/model/` | Thư mục lưu encodings |
| `TOLERANCE` | `float` | `0.6` | Ngưỡng matching (thấp = strict) |
| `DETECTION_CONFIDENCE` | `float` | `0.3` | YOLO conf cho live detection |
| `ENCODING_CONFIDENCE` | `float` | `0.5` | YOLO conf khi build encodings |
| `RECOGNITION_PADDING` | `int` | `15` | Padding pixel quanh face (recognize) |
| `ENCODING_PADDING` | `int` | `20` | Padding pixel quanh face (encode) |
| `MAX_FPS_SAMPLES` | `int` | `30` | Rolling window FPS |
| `LEARNING_DETECTION_THRESHOLD` | `float` | `0.4` | YOLO conf cho learning mode |
| `LEARNING_CAPTURE_THRESHOLD` | `float` | `0.25` | Min quality để capture |
| `MIN_FACE_SIZE` | `int` | `100` | Min face dimension (px) |
| `FRAME_MIN_LAPLACIAN` | `float` | `15` | Min Laplacian variance (frame extract) |
| `FRAME_MIN_FACE_PX` | `int` | `70` | Min face size (frame extract) |
| `FRAME_DETECT_CONF` | `float` | `0.15` | YOLO conf (frame extract) |
| `HIGH_CONFIDENCE_THRESHOLD` | `float` | `0.35` | YOLO conf (optimizer) |
| `OPTIMIZE_QUALITY_THRESHOLD` | `float` | `0.15` | Min quality (optimizer) |
| `MAX_ENCODINGS_PER_PERSON` | `int` | `50` | Max encodings per person sau clustering |
| `CLUSTERING_THRESHOLD` | `float` | `0.25` | Distance ngưỡng clustering (tăng từ 0.15 để giữ đa dạng góc/ánh sáng) |
| `FRAME_WIDTH` | `int` | `1280` | Camera width |
| `FRAME_HEIGHT` | `int` | `720` | Camera height |

---

## 2. Module: `src/core/face_utils.py`

### `extract_face_region(frame, x1, y1, x2, y2, padding) → Optional[np.ndarray]`

Crop vùng mặt từ frame với pixel padding.

| Parameter | Type | Mô tả |
|-----------|------|-------|
| `frame` | `np.ndarray` | BGR image |
| `x1, y1` | `int` | Top-left corner của bounding box |
| `x2, y2` | `int` | Bottom-right corner |
| `padding` | `int` | Pixel padding mở rộng mỗi cạnh |

**Returns:** `np.ndarray` (BGR crop) hoặc `None` nếu crop rỗng.

**Behavior:**
- Clamp tọaĐộ vào biên frame (không truy cập ngoài mảng)
- Trả `None` nếu vùng crop có diện tích = 0

---

## 3. Module: `src/core/encoder.py`

### `rebuild_encodings(dataset_path, model_path, encodings_path, logger) → Tuple[int, int]`

Đọc toàn bộ ảnh trong dataset, trích xuất face encodings, lưu file .pkl.

| Parameter | Type | Mô tả |
|-----------|------|-------|
| `dataset_path` | `str` | Thư mục gốc dataset (chứa các folder person) |
| `model_path` | `str` | Path đến YOLO .pt file |
| `encodings_path` | `str` | Path lưu file .pkl output |
| `logger` | `Optional[logging.Logger]` | Logger instance |

**Returns:** `(total_encodings, total_persons)` — số encoding và số person đã xử lý.

**Behavior:**
1. Load YOLO model
2. Duyệt mỗi person folder trong `dataset_path`
3. Với mỗi ảnh: YOLO detect → truyền bbox trực tiếp làm `known_face_locations` → `face_encodings()` → 128-d vector
4. Fallback: nếu YOLO không detect đủ confidence, chạy face_recognition trên toàn ảnh
5. Sau khi encode hết ảnh của một person → `_cluster_to_max()` → giới hạn `MAX_ENCODINGS_PER_PERSON=50`
6. Backup file .pkl cũ trước khi ghi mới
7. Lưu `{encodings: List[ndarray], names: List[str]}` vào pickle

---

## 4. Module: `src/core/frame_extractor.py`

### 4.1 Data Class: `ExtractedFrame`

```python
@dataclass
class ExtractedFrame:
    face_crop: np.ndarray      # BGR face crop (with padding)
    quality_score: float        # Combined quality [0, 1]
    source_frame_num: int       # Frame index in video
    timestamp: float            # Seconds from start
    detection_conf: float       # YOLO confidence (0.0 = HOG fallback)
```

### 4.2 Class: `FrameQualityChecker`

Đánh giá chất lượng từng frame.

#### `__init__(min_laplacian, min_brightness, max_brightness, min_face_size)`

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `min_laplacian` | `config.FRAME_MIN_LAPLACIAN` (15) | Laplacian variance tối thiểu |
| `min_brightness` | `0.10` | Brightness tối thiểu (0-1) |
| `max_brightness` | `0.92` | Brightness tối đa (0-1) |
| `min_face_size` | `config.FRAME_MIN_FACE_PX` (70) | Min cạnh ngắn nhất (px) |

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `is_sharp(face_crop)` | `np.ndarray` | `bool` | Laplacian var ≥ min_laplacian |
| `is_bright(face_crop)` | `np.ndarray` | `bool` | Brightness trong [min, max] |
| `is_large_enough(face_crop)` | `np.ndarray` | `bool` | min(h, w) ≥ min_face_size |
| `score(face_crop)` | `np.ndarray` | `float [0,1]` | Weighted: sharp 40% + bright 30% + size 30% |
| `breakdown(face_crop)` | `np.ndarray` | `dict` | Chi tiết từng metric |
| `passes_all(face_crop)` | `np.ndarray` | `bool` | Tất cả gate đều pass |
| `reject_reason(face_crop)` | `np.ndarray` | `Optional[str]` | `"too_small"`, `"too_dark"`, `"overexposed"`, `"blurry"`, hoặc `None` |

### 4.3 Class: `FrameDiversityFilter`

Lọc frame trùng lặp bằng SSIM.

#### `__init__(ssim_threshold=0.85, min_frame_gap=10)`

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `ssim_threshold` | `0.85` | SSIM ≥ ngưỡng → coi là trùng lặp |
| `min_frame_gap` | `10` | Khoảng cách frame tối thiểu giữa 2 lần lưu |

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `is_diverse(face_crop, frame_num)` | `ndarray, int` | `bool` | Frame đủ khác biệt? |
| `accept(face_crop, frame_num)` | `ndarray, int` | `None` | Ghi nhận frame đã lưu |
| `reset()` | — | `None` | Reset state cho session mới |

### 4.4 Class: `VideoFrameExtractor`

Pipeline chính trích xuất frame từ video.

#### `__init__(yolo_model, quality_checker, diversity_filter, frame_skip, detect_conf)`

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `yolo_model` | — | YOLO model instance |
| `quality_checker` | `FrameQualityChecker()` | Bộ lọc chất lượng |
| `diversity_filter` | `FrameDiversityFilter()` | Bộ lọc đa dạng |
| `frame_skip` | `3` | Bỏ qua mỗi N frame |
| `detect_conf` | `config.FRAME_DETECT_CONF` | YOLO confidence tối thiểu |

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `extract_from_file(video_path, max_frames)` | `str, int` | `Iterator[ExtractedFrame]` | Trích xuất từ video file |
| `extract_with_retry(video_path, target_frames)` | `str, int` | `Iterator[ExtractedFrame]` | Multi-pass với relaxed thresholds |
| `record_from_camera(camera_id, output_path)` | `int, str` | `Optional[str]` | Ghi video từ camera, trả path hoặc None |

---

## 5. Module: `src/recognizer.py`

### Class: `FaceRecognizer`

Hệ thống nhận diện real-time.

#### `__init__(dataset_path, model_path, tolerance, detection_confidence, encoding_confidence, padding, ...)`

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `dataset_path` | `config.DATASET_PATH` | Dataset folder |
| `model_path` | `config.MODEL_PATH` | YOLO model |
| `tolerance` | `0.6` | Face matching tolerance |
| `detection_confidence` | `0.3` | YOLO conf (live detect) |
| `encoding_confidence` | `0.5` | YOLO conf (build encodings) |
| `padding` | `15` | Padding pixel |
| `encodings_file` | `None` | Custom path cho pkl |

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `detect_faces_yolo(frame)` | `np.ndarray` | `List[(x1,y1,x2,y2,conf)]` | YOLO detect all faces |
| `recognize_face_in_region(frame, x1, y1, x2, y2)` | `ndarray, int×4` | `str` | Tên + confidence hoặc "Unknown". Dùng weighted voting: weight=(1-distance) |
| `draw_results(frame, detections, names)` | `ndarray, list, list` | `np.ndarray` | Vẽ bbox + label |
| `run_recognition(camera_id, frame_width, frame_height)` | `int, int, int` | `None` | Main loop nhận diện |

---

## 6. Module: `src/training_manager.py`

### 6.1 Data Classes

#### `PersonInfo`

```python
@dataclass
class PersonInfo:
    name: str               # Tên person (folder name)
    image_count: int        # Số ảnh gốc (không tính aug)
    augmented_count: int    # Số ảnh augmented
    folder_path: Path       # Path đến folder
    created_date: str       # ISO date format
```

#### `CollectionResult`

```python
@dataclass
class CollectionResult:
    person_name: str
    frames_processed: int               # Tổng frame đã xử lý
    frames_saved: int                   # Frame được lưu
    frames_rejected: int                # Frame bị bỏ qua
    reject_reasons: Dict[str, int]      # {"blurry": 12, "no_face": 45, ...}
    duration_seconds: float
    avg_quality: float                  # [0, 1]
```

#### `ExtractionSettings`

```python
@dataclass
class ExtractionSettings:
    min_laplacian: float = 15
    min_face_px: int = 70
    detect_conf: float = 0.15
    ssim_threshold: float = 0.85
    min_frame_gap: int = 10
    frame_skip: int = 3
```

| Method | Mô tả |
|--------|-------|
| `save(path)` | Lưu settings ra JSON |
| `load(path)` | Đọc settings từ JSON (class method) |

**Presets:**
- "Thoai mai": min_laplacian=5, min_face_px=50, detect_conf=0.08, ssim=0.93, gap=5, skip=2
- "Binh thuong": defaults
- "Khat khe": min_laplacian=40, min_face_px=120, detect_conf=0.30, ssim=0.78, gap=20, skip=6

### 6.2 Class: `PersonManager`

CRUD operations trên person folders.

#### `__init__(dataset_path=config.DATASET_PATH)`

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `list_persons()` | — | `List[PersonInfo]` | Danh sách persons với stats |
| `get_image_count(name)` | `str` | `int` | Đếm ảnh gốc |
| `get_augmented_count(name)` | `str` | `int` | Đếm ảnh augmented |
| `exists(name)` | `str` | `bool` | Person folder tồn tại? |
| `create_person(name)` | `str` | `Path` | Tạo folder (raise ValueError nếu tên không hợp lệ) |
| `delete_person(name, rebuild)` | `str, bool` | `bool` | Xóa folder + rebuild encodings |
| `reset_person(name, rebuild)` | `str, bool` | `int` | Xóa ảnh (giữ folder), trả số ảnh đã xóa |

### 6.3 Class: `DataCollectionSession`

Orchestrates một phiên thu thập dữ liệu.

#### `__init__(person_name, extractor, dataset_path=config.DATASET_PATH)`

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `run_from_file(video_path, max_frames)` | `str, int` | `CollectionResult` | Thu thập từ video file |
| `run_from_camera(camera_id, target_frames, raw_video_dir)` | `int, int, Optional[str]` | `CollectionResult` | Thu thập từ camera (ghi video → extract) |

### 6.4 Class: `InteractiveCLI`

Giao diện dòng lệnh tương tác.

#### Methods (internal flows)

| Method | Mô tả |
|--------|-------|
| `run()` | Main loop menu |
| `_collect_data_flow()` | Flow thu thập dữ liệu |
| `_manage_persons_flow()` | Flow quản lý persons |
| `_augment_flow()` | Flow augment dataset |
| `_rebuild_flow()` | Flow rebuild encodings |
| `_optimize_flow()` | Flow optimize encodings |
| `_settings_flow()` | Flow cài đặt tham số |

---

## 7. Module: `src/augment_dataset.py`

### Registry: `AUGMENTATIONS: Dict[str, callable]`

| Key | Function | Mô tả |
|-----|----------|-------|
| `blur_mild` | `_aug_blur_mild` | Gaussian blur 5×5 |
| `blur_heavy` | `_aug_blur_heavy` | Gaussian blur 21×21 |
| `motion_blur` | `_aug_motion_blur` | Horizontal motion blur |
| `low_light_mild` | `_aug_low_light_mild` | Brightness ×0.5 |
| `low_light_heavy` | `_aug_low_light_heavy` | Brightness ×0.25 |
| `noise` | `_aug_noise` | Gaussian noise σ=30 |
| `low_quality` | `_aug_low_quality` | JPEG quality=15 |
| `combined` | `_aug_combined` | Blur + low-light |

### Functions

| Function | Input | Output | Mô tả |
|----------|-------|--------|-------|
| `augment_person(person_name, dataset_path, show_preview, logger)` | `str, str, bool, Logger` | `Tuple[int, int]` | Augment all images for one person. Returns (generated, skipped) |
| `preview_augmentations(sample_image, person_name)` | `ndarray, str` | `bool` | Show preview collage, return True if confirmed |

---

## 8. Module: `src/optimize_encodings.py`

### Class: `FaceEncodingOptimizer`

#### `__init__(dataset_path, model_path)`

#### Methods

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `load_existing_encodings()` | — | `Tuple[List[ndarray], List[str]]` | Load pkl |
| `extract_high_quality_faces(image_path)` | `str` | `List[Tuple[ndarray, float]]` | Extract encoding + quality từ 1 ảnh |
| `cluster_similar_encodings(encodings, qualities)` | `List, List` | `List[int]` | Trả về indices encoding được chọn |
| `optimize_person_encodings(person_name)` | `str` | `Tuple[List, int, int]` | Optimize 1 person |
| `optimize_all_encodings()` | — | `bool` | Optimize toàn bộ dataset |
| `benchmark_performance()` | — | `None` | Benchmark trước/sau optimize |

---

## 9. Module: `src/benchmark_models.py`

### Data Classes

```python
@dataclass
class FrameResult:
    detected: bool
    conf: float
    face_px: int
    n_faces: int
    ms: float

@dataclass
class ModelReport:
    model_name: str
    model_size_mb: float
    status: str          # "ok" | "error" | "incompatible"
    fps: float
    avg_ms: float
    p95_ms: float
    detect_rate: float   # % frames with face
    avg_conf: float
    avg_face_px: float
```

### Functions

| Function | Input | Output | Mô tả |
|----------|-------|--------|-------|
| `sample_frames(video_paths, max_frames, sample_every)` | `List[Path], int, int` | `List[ndarray]` | Trích xuất frames từ videos |
| `run_model(model_path, frames)` | `Path, List[ndarray]` | `ModelReport` | Benchmark 1 model |
| `save_samples(reports, frames, out_dir)` | `List, List, Path` | `None` | Lưu ảnh minh họa detection |

---

## 10. Test Suite (`tests/`)

### Fixtures (conftest.py)

| Fixture | Mô tả |
|---------|-------|
| `synthetic_face_image` | 200×200 BGR image với gradient + circle face |
| `blurry_image` | Blurred version (Gaussian 31×31) |
| `dark_image` | Very dark (×0.05) |
| `tiny_face_image` | 50×50 image |
| `tmp_dataset` | Temp dir với 2 persons × 3 images |
| `synthetic_video` | 30-frame MP4 từ synthetic_face_image |

### Test Files

| File | Tests | Covers |
|------|-------|--------|
| `test_frame_quality.py` | 11 | `FrameQualityChecker` — sharp/bright/size/score |
| `test_person_manager.py` | 13 | `PersonManager` — CRUD |
| `test_frame_extractor.py` | 8 | `FrameDiversityFilter`, `VideoFrameExtractor` |
| `test_integration.py` | 5 | Full pipeline: create → collect → verify |

### Run Tests

```bash
cd /home/dkhai/workspace
pytest tests/ -v
```
