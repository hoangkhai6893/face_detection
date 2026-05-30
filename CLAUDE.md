# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `workspace/` (the working directory inside Docker / Dev Container).

```bash
# Real-time face recognition
python src/recognizer.py
python src/recognizer.py --camera 0 --tolerance 0.6 --model src/yolo/yolov12s-face.pt --verbose

# Collect training data / manage persons (interactive CLI menu)
python src/training_manager.py

# Augment existing dataset (8 variations per image)
python src/augment_dataset.py
python src/augment_dataset.py --person Khai

# Rebuild encoding database from dataset
# (done automatically via training_manager.py menu, or call rebuild_encodings() from core/encoder.py)

# Optimize encodings via clustering (~90% reduction)
python src/optimize_encodings.py

# Export YOLO .pt → .onnx for 1.5-2x CPU speedup (run once)
python src/export_onnx.py
python src/export_onnx.py --model src/yolo/yolov11s-face.pt --imgsz 416

# Run tests (from workspace/)
pytest tests/ -v
pytest tests/test_frame_quality.py -v       # quality filter unit tests
pytest tests/test_person_manager.py -v      # CRUD person tests
pytest tests/test_frame_extractor.py -v     # video frame extraction tests
pytest tests/test_integration.py -v         # integration tests

# Benchmark YOLO models against a video file
python src/benchmark_models.py
```

## Architecture

### Two-library hybrid pipeline

The system combines **YOLO** (face detection) and **face_recognition/dlib** (face identification). They are never interchangeable — YOLO locates bounding boxes; dlib computes 128-d encoding vectors for identity matching.

Real-time recognition loop (`src/recognizer.py → FaceRecognizer`):
1. A background capture thread feeds frames into a `Queue(maxsize=1)` — main loop always gets the freshest frame, never queues stale ones.
2. Frame is downscaled to `YOLO_INPUT_WIDTH=416` before YOLO inference; bounding boxes are scaled back up.
3. BGR→RGB conversion happens once per frame, not once per face.
4. An IoU-based face cache (`RECOGNITION_INTERVAL=10`) skips dlib encoding for faces seen within the last 10 frames. This eliminates ~80-93% of expensive dlib calls.
5. When recognition runs, the YOLO bbox is passed directly as `known_face_locations` to `face_recognition.face_encodings()`, bypassing dlib's internal HOG detector (~30-50% faster).
6. Match decision: `face_distance() ≤ TOLERANCE (0.6)` → known person; otherwise → "Unknown".

### Data flow for adding a new person

```
Camera/Video recording
    → VideoFrameExtractor (core/frame_extractor.py)
        → FrameQualityChecker (sharpness/brightness/size gates)
        → FrameDiversityFilter (SSIM deduplication)
    → family_images/<Person>/*.jpg
    → (optional) augment_dataset.py → 8 augmented variants
    → rebuild_encodings() (core/encoder.py) → model/face_encodings_hybrid.pkl
    → (optional) optimize_encodings.py → clustering to remove duplicates
```

After any dataset change, `rebuild_encodings()` must be called before recognition sees the new person.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/config.py` | Single source of truth for all paths and tunable parameters |
| `src/recognizer.py` | `FaceRecognizer` class — real-time camera loop |
| `src/training_manager.py` | Interactive CLI: `PersonManager` (CRUD), `DataCollectionSession`, `ExtractionSettings` |
| `src/augment_dataset.py` | Generates 8 image variants per source image (blur, low-light, noise, JPEG compression) |
| `src/optimize_encodings.py` | `FaceEncodingOptimizer` — pairwise clustering to deduplicate encodings |
| `src/export_onnx.py` | One-time YOLO `.pt` → `.onnx` export; `config.py` auto-selects `.onnx` when present |
| `src/benchmark_models.py` | Benchmarks all YOLO models against a video and scores them |
| `src/core/face_utils.py` | `extract_face_region()` — padded crop with boundary clamping, shared across all modules |
| `src/core/encoder.py` | `rebuild_encodings()` — reads dataset, runs YOLO+dlib, writes `.pkl` |
| `src/core/frame_extractor.py` | `VideoFrameExtractor`, `FrameQualityChecker`, `FrameDiversityFilter` |

### Data stores

| Path | Format | Contents |
|---|---|---|
| `workspace/family_images/<Person>/` | JPEG files | Training images per person |
| `workspace/model/face_encodings_hybrid.pkl` | Pickle | `{encodings: List[ndarray], names: List[str]}` |
| `workspace/model/*_backup.pkl` | Pickle | Auto-backup before optimize/rebuild |
| `workspace/data/*.mp4` | MP4 | Raw video recordings |
| `workspace/extraction_settings.json` | JSON | Persisted extraction parameters |

### Key configuration (src/config.py)

All paths are resolved relative to `config.py`'s own location — no hardcoded absolute paths. Key tuning parameters:

| Parameter | Default | Effect |
|---|---|---|
| `TOLERANCE` | 0.6 | Matching strictness (lower = stricter) |
| `RECOGNITION_INTERVAL` | 10 | Frames between full dlib re-encodes (cache TTL) |
| `YOLO_INPUT_WIDTH` | 416 | Frame width before YOLO (0 = no resize) |
| `DETECTION_CONFIDENCE` | 0.3 | Min YOLO confidence for live detection |
| `FRAME_MIN_LAPLACIAN` | 15 | Min Laplacian variance for training frame sharpness |
| `CLUSTERING_THRESHOLD` | 0.15 | Max face distance to be considered a near-duplicate |
| `MAX_ENCODINGS_PER_PERSON` | 30 | Max kept per person after clustering |

### ONNX model auto-selection

`config.py` checks for `yolov11n-face.onnx` at startup; if present it is used over the `.pt` file. Run `python src/export_onnx.py` once to generate it. ONNX Runtime is ~1.5–2× faster than PyTorch on CPU with no code changes elsewhere.

### Test layout

Tests live in `workspace/tests/`. `conftest.py` adds `src/` to `sys.path` and provides shared fixtures (synthetic images, blurry/dark/overexposed variants, `tmp_dataset`, `synthetic_video`). Tests do not require a camera or real face images.

### Import paths

Scripts in `src/` import `config` and `core.*` without any `src.` prefix because they are run with `workspace/src/` as their working context. Tests add `src/` to `sys.path` via `conftest.py`. When running pytest, `cd workspace` first.
