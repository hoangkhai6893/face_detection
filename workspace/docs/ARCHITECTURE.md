# Kiến Trúc Hệ Thống — Family Face Recognition

**Phiên bản:** 2.0
**Ngày:** 2026-03-29

---

## 1. Tổng Quan Hệ Thống

Hệ thống nhận diện khuôn mặt gia đình kết hợp **YOLO** (face detection) và **face_recognition / dlib** (face identification). Hệ thống gồm 4 quy trình chính:

| Quy trình | Entry Point | Mô tả |
|-----------|-------------|-------|
| **Nhận diện real-time** | `src/recognizer.py` | Camera → detect → identify → hiển thị |
| **Thu thập dữ liệu** | `src/training_manager.py` | Camera/Video → frame selection → lưu ảnh |
| **Augment dataset** | `src/augment_dataset.py` | Ảnh gốc → 8 biến thể → tăng dataset |
| **Tối ưu encodings** | `src/optimize_encodings.py` | Clustering → giảm encoding trùng lặp |

---

## 2. Sơ Đồ Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INTERACTIVE CLI                                │
│                     training_manager.py (Entry Point)                   │
│                                                                         │
│   ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────┐     │
│   │ PersonManager│  │DataCollectionSession│ │  ExtractionSettings   │     │
│   │ CRUD persons │  │ run_from_file()  │  │  Presets / tuning     │     │
│   │              │  │ run_from_camera()│  │  → extraction_settings│     │
│   └─────────────┘  └───────┬──────────┘  └───────────────────────┘     │
│                            │                                            │
│   ┌─────────────┐  ┌──────▼──────────────────────────────────────┐     │
│   │ augment_    │  │          VideoFrameExtractor                 │     │
│   │ dataset.py  │  │                                            │     │
│   │ (8 biến thể)│  │  ┌──────────────────┐  ┌────────────────┐  │     │
│   └─────────────┘  │  │FrameQualityChecker│  │FrameDiversity  │  │     │
│                     │  │ - is_sharp()     │  │Filter          │  │     │
│   ┌─────────────┐  │  │ - is_bright()    │  │ - SSIM dedup   │  │     │
│   │ optimize_   │  │  │ - is_large()     │  │ - frame gap    │  │     │
│   │ encodings.py│  │  │ - score()        │  └────────────────┘  │     │
│   │ (clustering)│  │  └──────────────────┘                       │     │
│   └─────────────┘  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐    ┌──────────────────────────┐
│   RECOGNIZER     │    │      DATA STORES          │
│ recognizer.py    │    │                           │
│                  │    │ family_images/<Person>/    │
│ YOLO detect      │    │ ├── img_001_q0.85.jpg    │
│ → face crop      │    │ └── img_002_aug_blur.jpg │
│ → 128d encoding  │    │                           │
│ → match DB       │    │ model/                    │
│ → label + bbox   │    │ └── face_encodings_       │
│                  │    │     hybrid.pkl             │
└──────────────────┘    └──────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│              CORE UTILITIES              │
│  src/core/                               │
│                                          │
│  face_utils.py    — crop face region     │
│  encoder.py       — rebuild encodings    │
│  frame_extractor.py — video extraction   │
└──────────────────────────────────────────┘
```

---

## 3. Thành Phần Hệ Thống

### 3.1 Core Layer (`src/core/`)

| Module | Class / Function | Trách nhiệm |
|--------|-----------------|-------------|
| `face_utils.py` | `extract_face_region()` | Crop vùng mặt từ frame với padding, clamp biên |
| `encoder.py` | `rebuild_encodings()` | Đọc toàn bộ dataset → YOLO detect → face_recognition encode → lưu .pkl |
| `frame_extractor.py` | `FrameQualityChecker` | Đánh giá chất lượng frame: độ nét, độ sáng, kích thước |
| `frame_extractor.py` | `FrameDiversityFilter` | Lọc frame trùng lặp bằng SSIM + khoảng cách temporal |
| `frame_extractor.py` | `VideoFrameExtractor` | Pipeline chính: đọc video → detect → filter → yield frame |
| `frame_extractor.py` | `ExtractedFrame` | Dataclass chứa face_crop, quality_score, metadata |

### 3.2 Application Layer (`src/`)

| Module | Class / Function | Trách nhiệm |
|--------|-----------------|-------------|
| `config.py` | Constants | Toàn bộ tham số hệ thống (paths, thresholds, parameters) |
| `recognizer.py` | `FaceRecognizer` | Nhận diện real-time: camera → YOLO detect → encode → match |
| `training_manager.py` | `InteractiveCLI` | Giao diện dòng lệnh tương tác (questionary) |
| `training_manager.py` | `PersonManager` | CRUD người: tạo, xem, xóa, reset person folders |
| `training_manager.py` | `DataCollectionSession` | Phiên thu thập: từ camera hoặc video file |
| `training_manager.py` | `ExtractionSettings` | Quản lý presets và tuning tham số trích xuất |
| `augment_dataset.py` | `augment_person()` | Tạo 8 biến thể ảnh (blur, dark, noise, ...) |
| `optimize_encodings.py` | `FaceEncodingOptimizer` | Clustering encodings, giảm số lượng, benchmark |
| `benchmark_models.py` | `run_model()` | Benchmark các YOLO model trên video |

### 3.3 Data Layer

| Path | Format | Nội dung |
|------|--------|----------|
| `family_images/<Person>/` | JPG files | Ảnh training per person (gốc + augmented) |
| `model/face_encodings_hybrid.pkl` | Pickle | `{encodings: List[ndarray], names: List[str]}` |
| `model/*_backup.pkl` | Pickle | Backup encodings trước khi optimize/rebuild |
| `data/*.mp4` | MP4 | Video raw recordings |
| `extraction_settings.json` | JSON | Persistent extraction parameters |

### 3.4 Models

| File | Kích thước | Đặc điểm |
|------|-----------|-----------|
| `yolov11n-face.pt` | ~5 MB | Nhanh nhất (~17 FPS), phù hợp real-time |
| `yolov11s-face.pt` | ~20 MB | Cân bằng tốc độ/độ chính xác |
| `yolov12s-face.pt` | ~20 MB | Mới nhất, chính xác hơn (~7 FPS) |

---

## 4. Nhận Diện Real-Time Pipeline

```
Camera Frame (1280×720)
        │
        ▼
┌───────────────────────┐
│   YOLO Detection       │  detect_conf = 0.3
│   Trả về bbox + conf  │  Tìm tất cả face trong frame
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   extract_face_region │  padding = 15px
│   Crop vùng mặt       │  Clamp vào biên frame
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   face_recognition     │
│   .face_encodings()    │  BGR → RGB → 128-d vector
│   (dlib HOG model)    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Matching             │
│   face_distance()      │  So sánh với DB encodings
│   tolerance = 0.6      │  distance ≤ 0.6 → KNOWN
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Visualization        │
│                        │
│  Green  = Known person │
│  Orange = Unknown      │
│  Red    = Error        │
│  Yellow = No training  │
│                        │
│  Display: FPS, names,  │
│  confidence scores     │
└───────────────────────┘
```

---

## 5. Thu Thập Dữ Liệu Pipeline

```
[User chọn: Camera hoặc Video file]
            │
     ┌──────┴──────┐
     ▼             ▼
 Camera          Video File
 record          read
     │             │
     └──────┬──────┘
            │
            ▼
┌───────────────────────────────────────────────┐
│  VideoFrameExtractor.extract_from_file()       │
│                                                │
│  For each frame (skip every N frames):         │
│    1. YOLO detect face                         │
│       └─ No face → SKIP                        │
│    2. extract_face_region() + padding          │
│    3. FrameQualityChecker:                     │
│       ├─ Laplacian var < min → SKIP (blurry)   │
│       ├─ brightness out of range → SKIP        │
│       └─ size < min → SKIP (too small)         │
│    4. FrameDiversityFilter:                    │
│       ├─ SSIM ≥ threshold → SKIP (duplicate)   │
│       └─ gap < min_frame_gap → SKIP (too soon) │
│    5. YIELD ExtractedFrame                     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   _save_frame()       │
            │                       │
            │   <Name>_<ts>_q<sc>.jpg│
            │   → family_images/    │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   rebuild_encodings() │
            │   → face_encodings_   │
            │     hybrid.pkl        │
            └───────────────────────┘
```

### Retry Logic (Camera mode)

```
Pass 1: ssim=0.85, gap=15  (strict)
   │  Nếu chưa đủ target_frames:
Pass 2: ssim=0.92, gap=5   (relaxed)
   │  Nếu chưa đủ:
Pass 3: ssim=0.99, gap=1   (minimal filter)
```

---

## 6. Encoding Rebuild Pipeline

```
family_images/
├── Person_A/
│   ├── img1.jpg, img2.jpg, ...
│   └── img1_aug_blur.jpg, ...
├── Person_B/
│   └── ...
        │
        ▼
┌──────────────────────────────────┐
│  For each person folder:         │
│                                  │
│  For each image (.jpg/.png):     │
│    1. cv2.imread()               │
│    2. YOLO detect → best box     │
│       ├─ conf ≥ 0.5 → crop      │
│       └─ conf < 0.5 → fallback  │
│    3. face_recognition           │
│       .face_encodings(crop)      │
│       → 128-d vector             │
│    4. Append to list             │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Backup old .pkl                 │
│  Save new .pkl:                  │
│  {                               │
│    encodings: [ndarray, ...],    │
│    names: ["Alice", "Bob", ...]  │
│  }                               │
└──────────────────────────────────┘
```

---

## 7. Encoding Optimization Pipeline

```
face_encodings_hybrid.pkl (200-400 encodings)
        │
        ▼
┌──────────────────────────────────┐
│  For each person:                │
│                                  │
│  1. Re-extract encodings from    │
│     images (YOLO + quality)      │
│                                  │
│  2. Pairwise distance matrix     │
│                                  │
│  3. Clustering:                  │
│     distance < 0.15 → same group │
│                                  │
│  4. Select representative:       │
│     closest to centroid          │
│                                  │
│  5. If > 30 per person:          │
│     quality-diverse sampling     │
└──────────────┬───────────────────┘
               │
               ▼
face_encodings_hybrid.pkl (50-90 encodings)
   ~3-5x faster recognition
```

---

## 8. Dataset Augmentation Pipeline

```
family_images/<Person>/*.jpg (original)
        │
        ▼
┌──────────────────────────────────┐
│  8 transformations per image:    │
│                                  │
│  1. blur_mild     (5×5 Gauss)   │
│  2. blur_heavy    (21×21 Gauss) │
│  3. motion_blur   (horizontal)  │
│  4. low_light_mild (×0.5)       │
│  5. low_light_heavy (×0.25)     │
│  6. noise         (σ=30)        │
│  7. low_quality   (JPEG q=15)   │
│  8. combined      (blur+dark)   │
│                                  │
│  Output: <stem>_aug_<type>.jpg   │
│  Idempotent: skip if exists      │
└──────────────┬───────────────────┘
               │
               ▼
   Cần rebuild_encodings() sau đó
```

---

## 9. Benchmark Pipeline

```
src/yolo/*.pt  ×  data/*.mp4
        │
        ▼
┌──────────────────────────────────┐
│  1. sample_frames()              │
│     Trích xuất frames từ video   │
│     (chia đều cho tất cả model)  │
│                                  │
│  2. For each model:              │
│     - Warm-up 5 frames           │
│     - Run inference              │
│     - Record: FPS, detect%,      │
│       conf, face_px, p95_ms     │
│                                  │
│  3. Scoring:                     │
│     FPS 40% + Detect 35% +      │
│     Conf 15% + FaceSize 10%     │
│                                  │
│  4. Recommendation:              │
│     Best overall / Real-time /   │
│     Best accuracy                │
└──────────────────────────────────┘
```

---

## 10. Quyết Định Thiết Kế

### Tại sao YOLO + face_recognition (hybrid)?

| Approach | Ưu điểm | Nhược điểm |
|----------|----------|-------------|
| Dlib HOG only | Không cần model file | Chậm, không detect được mặt xa |
| YOLO only | Nhanh, detect tốt | Không có encoding/matching |
| **YOLO + face_recognition** | **Nhanh + chính xác** | Cần cả 2 thư viện |

### Tại sao SSIM thay vì pixel difference?

- Pixel diff: nhạy cảm với noise, shift nhỏ
- SSIM: đánh giá cấu trúc tổng thể, bền vững hơn

### Tại sao multi-pass retry (camera mode)?

- Pass 1 strict → đảm bảo chất lượng cao
- Pass 2-3 relaxed → đảm bảo đủ số lượng frame khi video ngắn

### Tại sao pickle thay vì SQLite/JSON?

- face_recognition trả về numpy arrays → pickle serialize hiệu quả nhất
- Dataset nhỏ (vài trăm encoding) → không cần database
- Đơn giản, nhanh, ít dependency
