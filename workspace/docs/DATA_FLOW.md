# Luồng Dữ Liệu — Family Face Recognition System

**Phiên bản:** 2.0
**Ngày:** 2026-03-29

---

## 1. Tổng Quan Luồng Dữ Liệu

Hệ thống có 4 luồng dữ liệu chính:

```
                    ┌─────────────────────────┐
                    │      CAMERA / VIDEO      │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
     │  THU THẬP      │ │  NHẬN DIỆN    │ │  BENCHMARK     │
     │  DỮ LIỆU       │ │  REAL-TIME    │ │  MODELS        │
     └───────┬────────┘ └───────┬───────┘ └────────────────┘
             │                  │
             ▼                  ▼
     ┌────────────────┐ ┌───────────────┐
     │  family_images/ │ │  bbox + name  │
     │  <Person>/*.jpg │ │  trên màn hình│
     └───────┬────────┘ └───────────────┘
             │
             ▼
     ┌────────────────┐
     │  AUGMENT        │
     │  (+8 biến thể)  │
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │  REBUILD /      │
     │  OPTIMIZE       │
     │  ENCODINGS      │
     └───────┬────────┘
             │
             ▼
     ┌────────────────┐
     │ face_encodings_ │
     │ hybrid.pkl      │
     └────────────────┘
```

---

## 2. Luồng Thu Thập Dữ Liệu (Data Collection)

### 2.1 Từ Camera

```
User                    InteractiveCLI              VideoFrameExtractor
 │                           │                            │
 │─ Chọn "Thu thập" ───────▶│                            │
 │─ Chọn person ────────────▶│                            │
 │─ Chọn "Camera" ──────────▶│                            │
 │─ Nhập target_frames ─────▶│                            │
 │                           │─ run_from_camera() ───────▶│
 │                           │                            │
 │                           │                    ┌───────┴────────┐
 │                           │                    │ record_from_   │
 │◀── Preview window ────────│                    │ camera()       │
 │─ SPACE = bắt đầu ghi ────▶│                    │                │
 │                           │                    │ Phase 1:       │
 │◀── REC indicator ─────────│                    │ Preview + wait │
 │                           │                    │                │
 │                           │                    │ Phase 2:       │
 │                           │                    │ Record raw .mp4│
 │─ SPACE/Q/ESC = dừng ─────▶│                    │                │
 │                           │                    └───────┬────────┘
 │                           │◀── recorded_path ──────────┘
 │                           │
 │                           │─ extract_with_retry() ───▶│
 │                           │                            │
 │                           │                    ┌───────┴────────┐
 │                           │                    │ Pass 1: strict │
 │                           │                    │ (ssim=0.85)    │
 │                           │                    │                │
 │                           │                    │ Pass 2: relaxed│
 │                           │                    │ (ssim=0.92)    │
 │                           │                    │                │
 │                           │                    │ Pass 3: minimal│
 │                           │                    │ (ssim=0.99)    │
 │                           │                    └───────┬────────┘
 │                           │                            │
 │                           │◀── Iterator[ExtractedFrame]┘
 │                           │
 │                           │─ _save_frame() × N
 │                           │  → family_images/<Person>/
 │                           │
 │                           │─ rebuild_encodings()
 │                           │  → face_encodings_hybrid.pkl
 │                           │
 │◀── Summary ───────────────│
 │   frames_saved, avg_quality, duration
```

### 2.2 Từ Video File

```
User                    InteractiveCLI              VideoFrameExtractor
 │                           │                            │
 │─ Chọn "Video file" ─────▶│                            │
 │─ Nhập đường dẫn ─────────▶│                            │
 │─ Nhập max_frames ────────▶│                            │
 │                           │─ run_from_file(path, max)─▶│
 │                           │                            │
 │                           │                    ┌───────┴────────┐
 │                           │                    │ extract_from_  │
 │                           │                    │ file()         │
 │                           │                    │                │
 │                           │                    │ For each frame:│
 │                           │                    │  skip % N != 0 │
 │                           │                    │    → continue  │
 │                           │                    │  YOLO detect   │
 │                           │                    │    → no face?  │
 │                           │                    │      → skip    │
 │                           │                    │  crop + padding│
 │                           │                    │  quality check │
 │                           │                    │    → fail?     │
 │                           │                    │      → skip    │
 │                           │                    │  diversity chk │
 │                           │                    │    → dup?      │
 │                           │                    │      → skip    │
 │                           │                    │  YIELD frame   │
 │                           │                    └───────┬────────┘
 │                           │                            │
 │                           │◀── ExtractedFrame ─────────┘
 │                           │─ _save_frame()
 │                           │  → family_images/<Person>/
 │                           │─ rebuild_encodings()
 │◀── Summary ───────────────│
```

---

## 3. Luồng Nhận Diện Real-Time

```
Camera              FaceRecognizer                   Display
 │                       │                              │
 │──── frame ──────────▶│                              │
 │                       │                              │
 │                       │── detect_faces_yolo() ──┐    │
 │                       │                         │    │
 │                       │   YOLO inference        │    │
 │                       │   confidence > 0.3?     │    │
 │                       │   → bbox (x1,y1,x2,y2) │    │
 │                       │◀────────────────────────┘    │
 │                       │                              │
 │                       │── For each detection: ──┐    │
 │                       │                         │    │
 │                       │   extract_face_region() │    │
 │                       │   padding = 15px        │    │
 │                       │                         │    │
 │                       │   face_recognition      │    │
 │                       │   .face_encodings()     │    │
 │                       │   → 128-d vector        │    │
 │                       │                         │    │
 │                       │   face_distance()       │    │
 │                       │   vs known_encodings    │    │
 │                       │                         │    │
 │                       │   min_dist ≤ 0.6?       │    │
 │                       │    YES → "Name (conf)"  │    │
 │                       │    NO  → "Unknown"      │    │
 │                       │    empty → "No Data"    │    │
 │                       │◀────────────────────────┘    │
 │                       │                              │
 │                       │── draw_results() ───────────▶│
 │                       │   rectangle + label          │
 │                       │   FPS, face count            │
 │                       │                              │
 │                       │◀──── 'q' = quit ────────────│
```

### Màu sắc kết quả:

| Màu | BGR | Ý nghĩa |
|-----|-----|---------|
| Xanh lá | `(0, 255, 0)` | Đã nhận diện (Known person) |
| Cam | `(0, 165, 255)` | Không xác định (Unknown) |
| Đỏ | `(0, 0, 255)` | Lỗi / Không detect được |
| Vàng | `(0, 255, 255)` | Chưa có dữ liệu training |

---

## 4. Luồng Augment Dataset

```
family_images/<Person>/
├── img_001.jpg        ← original
├── img_002.jpg        ← original
└── ...
        │
        ▼
┌─────────────────────────────────────────┐
│  augment_person(person_name)            │
│                                         │
│  For each original image:               │
│    For each augmentation type:          │
│      dest = <stem>_aug_<type>.jpg       │
│      if exists → skip (idempotent)      │
│      else → apply transform → save      │
│                                         │
│  8 augmentation types:                  │
│  ┌──────────────┬───────────────────┐   │
│  │ blur_mild    │ Gaussian 5×5      │   │
│  │ blur_heavy   │ Gaussian 21×21    │   │
│  │ motion_blur  │ Horizontal kernel │   │
│  │ low_light_mild│ ×0.5 brightness  │   │
│  │ low_light_heavy│ ×0.25 brightness│   │
│  │ noise        │ σ=30 Gaussian     │   │
│  │ low_quality  │ JPEG quality=15   │   │
│  │ combined     │ blur + dark       │   │
│  └──────────────┴───────────────────┘   │
└─────────────────────────────────────────┘
        │
        ▼
family_images/<Person>/
├── img_001.jpg
├── img_001_aug_blur_mild.jpg
├── img_001_aug_blur_heavy.jpg
├── img_001_aug_motion_blur.jpg
├── img_001_aug_low_light_mild.jpg
├── img_001_aug_low_light_heavy.jpg
├── img_001_aug_noise.jpg
├── img_001_aug_low_quality.jpg
├── img_001_aug_combined.jpg
└── ...
        │
        ▼
  Cần rebuild_encodings() để cập nhật
```

---

## 5. Luồng Rebuild Encodings

```
family_images/
├── Alice/ (50 ảnh)
├── Bob/   (30 ảnh)
└── Carol/ (40 ảnh)
        │
        ▼
┌───────────────────────────────────────┐
│  rebuild_encodings()                  │
│                                       │
│  model = YOLO(config.MODEL_PATH)     │
│                                       │
│  For each person folder:              │
│    For each image:                    │
│      img = cv2.imread()              │
│      │                                │
│      ├─ YOLO detect                   │
│      │  └─ best box conf ≥ 0.5?      │
│      │     YES → crop + pad           │
│      │     └─ face_recognition        │
│      │        .face_encodings(crop)   │
│      │        → 128-d vector          │
│      │                                │
│      └─ Fallback: face_recognition    │
│         .face_encodings(full_img)     │
│         → 128-d vector                │
│                                       │
│  Result: all_encodings[], all_names[] │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Backup old pkl → *_backup.pkl        │
│  Save new pkl:                        │
│  {                                    │
│    "encodings": [ndarray × 120],      │
│    "names": ["Alice", "Bob", ...]     │
│  }                                    │
│  → model/face_encodings_hybrid.pkl    │
└───────────────────────────────────────┘
```

---

## 6. Luồng Optimize Encodings

```
model/face_encodings_hybrid.pkl (120 encodings)
        │
        ▼
┌───────────────────────────────────────┐
│  FaceEncodingOptimizer                │
│                                       │
│  For each person:                     │
│    1. Re-extract encodings from       │
│       images with quality scores      │
│                                       │
│    2. Filter: quality > 0.15          │
│                                       │
│    3. Pairwise distance matrix:       │
│       dist[i][j] = face_distance()   │
│                                       │
│    4. Clustering:                     │
│       if dist[i][j] < 0.15:          │
│         → same cluster                │
│       Select centroid representative  │
│                                       │
│    5. If still > 30 per person:       │
│       Quality-diverse sampling:       │
│       sort by quality, pick evenly    │
│       → giữ cả high + low quality     │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Backup old → *_backup.pkl            │
│  Save optimized:                      │
│  {                                    │
│    "encodings": [ndarray × 60],       │
│    "names": [...]                     │
│  }                                    │
│  → face_encodings_hybrid.pkl          │
│                                       │
│  Benchmark:                           │
│  100 comparisons:                     │
│    Before: ~0.5s                      │
│    After:  ~0.15s  (3-5x faster)     │
└───────────────────────────────────────┘
```

---

## 7. Format Dữ Liệu

### 7.1 Ảnh Training (family_images/<Person>/)

```
Naming convention:
  <PersonName>_<timestamp_ms>_q<score>.jpg     ← từ camera/video
  <PersonName>_<timestamp_ms>_conf<conf>.jpg   ← từ manual add
  <stem>_aug_<type>.jpg                        ← từ augment

Examples:
  Khai_1774709566173_q0.85.jpg
  Khai_1774709566200_q0.72.jpg
  Khai_1774709566173_aug_blur_mild.jpg
  MaiAnh_1774744977167_q0.91.jpg
```

### 7.2 Encodings File (face_encodings_hybrid.pkl)

```python
{
    "encodings": [
        np.ndarray(shape=(128,), dtype=float64),  # encoding person 1
        np.ndarray(shape=(128,), dtype=float64),  # encoding person 1
        np.ndarray(shape=(128,), dtype=float64),  # encoding person 2
        ...
    ],
    "names": [
        "Khai",
        "Khai",
        "MaiAnh",
        ...
    ]
}
# encodings[i] corresponds to names[i]
# One encoding per training image
```

### 7.3 Extraction Settings (extraction_settings.json)

```json
{
  "min_laplacian": 15,       // Độ nét tối thiểu (Laplacian variance)
  "min_face_px": 70,         // Kích thước mặt tối thiểu (px)
  "detect_conf": 0.15,       // YOLO confidence tối thiểu
  "ssim_threshold": 0.85,    // Ngưỡng SSIM chống trùng lặp
  "min_frame_gap": 10,       // Khoảng cách frame tối thiểu
  "frame_skip": 3            // Bỏ qua mỗi N frame
}
```

### 7.4 Raw Video (data/)

```
data/
├── Khai_1774709566173.mp4      ← ghi từ camera
├── Khai_1774742114257.mp4
├── Khai_1774773585539.mp4
└── MaiAnh_1774744977167.mp4

Naming: <PersonName>_<timestamp_ms>.mp4
Format: mp4v codec, native camera resolution
```

---

## 8. Mapping Giữa Config và Data Flow

| Config Parameter | Default | Dùng trong | Ý nghĩa |
|-----------------|---------|------------|---------|
| `DATASET_PATH` | `family_images/` | Tất cả | Thư mục dataset |
| `MODEL_PATH` | `yolo/yolov12s-face.pt` | Tất cả | YOLO model |
| `ENCODINGS_DIR` | `model/` | Encoder, Optimizer | Thư mục encodings |
| `TOLERANCE` | 0.6 | Recognizer | Ngưỡng nhận diện |
| `DETECTION_CONFIDENCE` | 0.3 | Recognizer | YOLO confidence (live) |
| `ENCODING_CONFIDENCE` | 0.5 | Encoder | YOLO confidence (training) |
| `FRAME_MIN_LAPLACIAN` | 15 | Frame Extractor | Độ nét tối thiểu |
| `FRAME_MIN_FACE_PX` | 70 | Frame Extractor | Kích thước mặt tối thiểu |
| `FRAME_DETECT_CONF` | 0.15 | Frame Extractor | YOLO confidence (extract) |
| `MAX_ENCODINGS_PER_PERSON` | 30 | Optimizer | Số encoding tối đa/person |
| `CLUSTERING_THRESHOLD` | 0.15 | Optimizer | Ngưỡng distance clustering |
