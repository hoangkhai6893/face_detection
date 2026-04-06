# Kiến Trúc Hệ Thống — Family Face Recognition

**Phiên bản:** 3.0
**Ngày cập nhật:** 2026-04-05

---

## 1. Tổng Quan Hệ Thống

Hệ thống nhận diện khuôn mặt gia đình kết hợp **YOLO** (face detection) và **face_recognition / dlib** (face identification). Hệ thống gồm 5 quy trình chính:

| Quy trình | Entry Point | Mô tả |
|-----------|-------------|-------|
| **Nhận diện real-time** | `src/recognizer.py` | Camera → detect → identify → hiển thị |
| **Thu thập dữ liệu** | `src/training_manager.py` | Camera/Video → frame selection → lưu ảnh |
| **Augment dataset** | `src/augment_dataset.py` | Ảnh gốc → 8 biến thể → tăng dataset |
| **Tối ưu encodings** | `src/optimize_encodings.py` | Clustering → giảm encoding trùng lặp |
| **Export ONNX** | `src/export_onnx.py` | YOLO .pt → .onnx (tăng tốc CPU ~1.5-2x) |

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
│   │ dataset.py  │  │                                             │     │
│   │ (8 biến thể)│  │  ┌──────────────────┐  ┌────────────────┐  │     │
│   └─────────────┘  │  │FrameQualityChecker│  │FrameDiversity  │  │     │
│                     │  │ - is_sharp()     │  │Filter          │  │     │
│   ┌─────────────┐  │  │ - is_bright()    │  │ - SSIM dedup   │  │     │
│   │ optimize_   │  │  │ - is_large()     │  │ - frame gap    │  │     │
│   │ encodings.py│  │  │ - score()        │  └────────────────┘  │     │
│   │ (clustering)│  │  └──────────────────┘                       │     │
│   └─────────────┘  └─────────────────────────────────────────────┘     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │ export_onnx.py                                              │       │
│   │ YOLO .pt → .onnx (chạy 1 lần, dùng ONNX Runtime sau đó)   │       │
│   └─────────────────────────────────────────────────────────────┘       │
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
| `export_onnx.py` | `export_to_onnx()` | Export YOLO .pt sang .onnx, benchmark tốc độ |
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

| File | Kích thước | Đặc điểm | Ghi chú |
|------|-----------|-----------|---------|
| `yolov11n-face.pt` | ~5 MB | Nhanh nhất, phù hợp real-time | Default |
| `yolov11n-face.onnx` | ~10 MB | ONNX version — nhanh hơn ~1.5-2x trên CPU | Tạo bằng export_onnx.py |
| `yolov11s-face.pt` | ~20 MB | Cân bằng tốc độ/độ chính xác | Dùng khi cần accuracy cao hơn |
| `yolov12s-face.pt` | ~20 MB | Mới nhất, chính xác nhất | Chậm hơn, dùng offline |

> **Auto-detect:** `config.py` tự động dùng `.onnx` nếu tồn tại, fallback về `.pt`.

---

## 4. Nhận Diện Real-Time Pipeline

### 4.1 Sơ đồ luồng

```
Camera Frame (1280×720)
        │
        ▼ [Capture Thread — decoupled I/O]
┌───────────────────────┐
│   Frame Queue         │  maxsize=1 — luôn lấy frame mới nhất
│   (threading.Queue)   │  Drop frame cũ nếu chưa xử lý kịp
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Resize Frame        │  1280 → 416px (YOLO_INPUT_WIDTH)
│   (cv2.resize)        │  Giảm ~1.5-2x thời gian YOLO inference
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────────┐
│   YOLO Detection                      │
│   yolov11n-face.onnx (ONNX Runtime)   │  detect_conf = 0.3
│   Trả về bbox + conf                  │  ~20-40ms/frame (CPU)
│   Scale bbox ngược lại ×(1280/416)    │  ← ~40-80ms nếu dùng .pt
└───────────────────────┬───────────────┘
            │
            ▼
┌───────────────────────┐
│   BGR → RGB           │  Chuyển đổi 1 lần/frame
│   (không phải/face)   │  Tiết kiệm so với convert per-face
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│   Face Cache Check (IoU-based)                    │
│                                                   │
│   RECOGNITION_INTERVAL = 10 frames               │
│   ├─ IoU > 0.4 với cached bbox → REUSE name      │  ~80-93% frames
│   └─ Cache miss / expired → run recognition      │  ~7-20% frames
└───────────────────────┬───────────────────────────┘
            │
  (khi cache miss)
            ▼
┌───────────────────────┐
│   face_recognition    │  Dùng YOLO bbox làm known_face_locations
│   .face_encodings()   │  → skip HOG detection của dlib (~30-50% faster)
│   → 128-d vector      │  ~20-50ms/khuôn mặt (CPU, dlib CNN)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Matching             │
│   face_distance()      │  NumPy vectorized vs known_encodings
│   tolerance = 0.6      │  distance ≤ 0.6 → KNOWN
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────────┐
│   Update Cache & Visualization        │
│                                       │
│  Xanh lá  = Known person             │
│  Cam      = Unknown                   │
│  Đỏ       = Error                    │
│  Vàng     = No training data         │
│                                       │
│  Overlay: FPS, Faces, Known, Encodings│
└───────────────────────────────────────┘
```

### 4.2 Tham số hiệu suất (config.py)

| Tham số | Giá trị | Tác dụng |
|---------|---------|----------|
| `YOLO_INPUT_WIDTH` | **416** | Resize trước YOLO — tăng tốc ~1.5-2x so với 640 |
| `RECOGNITION_INTERVAL` | **10** | Re-encode mỗi 10 frames — giảm 50% lần gọi dlib |
| `DETECTION_CONFIDENCE` | 0.3 | Ngưỡng confidence tối thiểu (thấp = bắt mặt xa hơn) |
| `RECOGNITION_PADDING` | 15px | Padding quanh bbox khi encode |
| `TOLERANCE` | 0.6 | Ngưỡng matching (thấp hơn = strict hơn) |

### 4.3 Hiệu suất ước tính (CPU, không GPU)

| Tình huống | FPS ước tính |
|-----------|-------------|
| PyTorch + 640px + interval=5 (cũ) | ~10-17 FPS |
| ONNX + 416px + interval=10 (mới) | ~22-30 FPS |
| Gain | **+50-80%** |

---

## 5. Thu Thập Dữ Liệu Pipeline

```
[User chọn: Camera hoặc Video file]
            │
     ┌──────┴──────┐
     ▼             ▼
 Camera          Video File
 record()        read
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
│       ├─ Laplacian var < 15 → SKIP (blurry)   │
│       ├─ brightness out of [0.10, 0.92] → SKIP│
│       └─ size < 70px → SKIP (too small)       │
│    4. FrameDiversityFilter:                    │
│       ├─ SSIM ≥ threshold → SKIP (duplicate)   │
│       └─ gap < min_frame_gap → SKIP (too soon) │
│    5. YIELD ExtractedFrame                     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   _save_frame()       │
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

### Retry Logic (multi-pass)

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
│   ├── img1.jpg, img2.jpg, ...       (original)
│   └── img1_aug_blur.jpg, ...        (augmented)
├── Person_B/
│   └── ...
        │
        ▼ rebuild_encodings() [core/encoder.py]
┌──────────────────────────────────┐
│  For each person folder:         │
│                                  │
│  For each image (.jpg/.png):     │
│    1. cv2.imread()               │
│    2. YOLO detect → best box     │
│       ├─ conf ≥ 0.5 → crop+enc  │
│       └─ conf < 0.5 → fallback  │
│           face_recognition full  │
│    3. → 128-d vector             │
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
        ▼ optimize_encodings.py
┌──────────────────────────────────┐
│  For each person:                │
│                                  │
│  1. Re-extract encodings từ      │
│     images (YOLO + quality)      │
│                                  │
│  2. Pairwise distance matrix     │
│     O(n²) — chỉ chạy offline    │
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
   Giữ diversity: high-quality + adverse-condition
```

---

## 8. Dataset Augmentation Pipeline

```
family_images/<Person>/*.jpg (original)
        │
        ▼ augment_dataset.py
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

## 9. ONNX Export Pipeline

```
src/yolo/yolov11n-face.pt
        │
        ▼ export_onnx.py (chạy 1 lần)
┌──────────────────────────────────┐
│  YOLO.export(format="onnx")      │
│    simplify=True  — xóa node thừa│
│    dynamic=False  — fixed shape  │
│    opset=17       — compat tốt   │
│    imgsz=416      — khớp config  │
└──────────────┬───────────────────┘
               │
               ▼
src/yolo/yolov11n-face.onnx
        │
        ▼ config.py (auto-detect)
┌──────────────────────────────────┐
│  MODEL_PATH = .onnx if exists    │
│            else .pt              │
│                                  │
│  YOLO("model.onnx")              │
│  → ultralytics dùng ONNX Runtime │
│  → không thay đổi code khác      │
└──────────────────────────────────┘

Speedup CPU: ~1.5-2x (PyTorch overhead removed)
```

---

## 10. Benchmark Pipeline

```
src/yolo/*.pt  ×  data/*.mp4
        │
        ▼ benchmark_models.py
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

## 11. Quyết Định Thiết Kế

### Tại sao YOLO + face_recognition (hybrid)?

| Approach | Ưu điểm | Nhược điểm |
|----------|----------|-------------|
| Dlib HOG only | Không cần model file | Chậm, không detect được mặt xa |
| YOLO only | Nhanh, detect tốt | Không có encoding/matching |
| **YOLO + face_recognition** | **Nhanh + chính xác** | Cần cả 2 thư viện |

### Tại sao không viết lại bằng C++?

Tất cả bottleneck thực sự đã là C++ bên dưới:
- `YOLO (ultralytics)` → PyTorch C++/CUDA kernel
- `face_recognition` → dlib C++ CNN
- `cv2` → OpenCV C++
- `numpy` → C + BLAS

Python thuần (cache logic, IoU, queue) chiếm **< 3% thời gian**. Viết lại C++ chỉ tăng ~0.5 FPS với 6-12 tuần công sức — không đáng.

### Tại sao ONNX thay vì TensorRT?

- TensorRT yêu cầu NVIDIA GPU
- ONNX Runtime hoạt động trên mọi CPU, nhẹ hơn PyTorch
- Ultralytics hỗ trợ `YOLO("model.onnx")` natively — zero code change

### Tại sao YOLO_INPUT_WIDTH = 416 thay vì 320?

- 320px: quá nhỏ → khuôn mặt ở khoảng cách vừa bị miss
- 640px: quá lớn → YOLO chậm
- **416px**: cân bằng tốt nhất cho khuôn mặt gần đến trung bình

### Tại sao RECOGNITION_INTERVAL = 10?

- Interval=5 tại 17FPS: lag ~0.3s (cũ)
- **Interval=10 tại 20FPS: lag ~0.5s** — không nhận thấy bằng mắt
- Interval=15: lag ~0.9s — nhận thấy được khi người mới bước vào frame

### Tại sao SSIM thay vì pixel difference?

- Pixel diff: nhạy cảm với noise, shift nhỏ
- SSIM: đánh giá cấu trúc tổng thể, bền vững hơn

### Tại sao multi-pass retry?

- Pass 1 strict → đảm bảo chất lượng cao
- Pass 2-3 relaxed → đảm bảo đủ số frame khi video ngắn

### Tại sao pickle thay vì SQLite/JSON?

- face_recognition trả về numpy arrays → pickle serialize hiệu quả nhất
- Dataset nhỏ (vài trăm encoding) → không cần database
- Đơn giản, nhanh, ít dependency
