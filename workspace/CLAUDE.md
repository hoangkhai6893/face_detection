# Family Face Recognition System — CLAUDE.md

> Loaded automatically at session start. Keep this file concise — it replaces codebase scanning.

## Stack
- **Detection:** YOLO v11/v12 (ONNX preferred, .pt fallback)
- **Recognition:** `face_recognition` (dlib) — 128-d embeddings
- **Storage:** pickle `.pkl` for encodings, JPG images in `family_images/<person>/`
- **Python 3.8+**, no web framework

## Project Layout

```
src/
  main.py                  # Entry: real-time recognition (camera)
  face_manager.py          # Entry: training / data management CLI
  config.py                # ALL thresholds and paths — edit here first
  core/
    recognizer.py          # RecognitionPipeline — main recognition loop
    encoder.py             # rebuild_encodings() — builds .pkl from dataset
    frame_extractor.py     # Extract quality frames from video
    frame_quality.py       # Quality scoring (Laplacian, face size)
    frame_diversity.py     # Diversity filtering (avoid near-duplicate frames)
    recognition_stabilizer.py  # Temporal smoothing across frames
    motion_guard.py        # Skip blurry frames from motion
    face_utils.py          # extract_face_region() — crop helper
  training/
    cli.py                 # Training CLI commands
    data_collection.py     # Camera / video data collection
    person_manager.py      # CRUD for persons in dataset
    extraction_settings.py # Per-run extraction config
  services/
    notification_worker.py # Background notification thread
    alert_manager.py       # Alert rules and dispatch
    event_logger.py        # Structured event logging
    device_dispatcher.py   # Multi-device output routing
model/                     # encodings.pkl lives here
family_images/             # Dataset: one folder per person
scripts/                   # Standalone utility scripts
tests/                     # pytest tests
docs/                      # API.md, ARCHITECTURE.md, DATA_FLOW.md, DESIGN.md
```

## Key Data Flow

```
Camera → YOLO detect → crop + padding → face_recognition encode
       → top-K vote against encodings.pkl → RecognitionStabilizer → display
```

Training flow:
```
video/camera → FrameExtractor (quality + diversity filter) → family_images/<person>/
→ rebuild_encodings() → _cluster_to_max() per person → model/encodings.pkl
```
Clustering (max `MAX_ENCODINGS_PER_PERSON=50`) applied inside `rebuild_encodings()` và `update_person_encodings()` — không cần bước optimize riêng.

## Important Config Values (src/config.py)

| Constant | Default | Purpose |
|---|---|---|
| `TOLERANCE` | 0.5 | Face match threshold — lower = stricter |
| `RECOGNITION_TOP_K` | 5 | Top-K voting window |
| `CONFUSION_MARGIN` | 0.10 | Min gap winner vs runner-up |
| `MAX_ENCODINGS_PER_PERSON` | 50 | After clustering optimization |
| `CLUSTERING_THRESHOLD` | 0.25 | Near-duplicate dedup distance |

## Common Tasks

| Task | Command |
|---|---|
| Real-time recognition | `python src/main.py` |
| Manage persons / collect data | `python src/face_manager.py` |
| Rebuild encodings | via `face_manager.py` → option "rebuild" |
| Run tests | `pytest tests/` |
| Export YOLO to ONNX | `python scripts/export_onnx.py` |

## Subdirectory Context (auto-loaded by Claude Code)

- [src/core/CLAUDE.md](src/core/CLAUDE.md) — recognition pipeline, encoding format, decision logic
- [src/training/CLAUDE.md](src/training/CLAUDE.md) — data collection, dataset layout, CLI
- [src/services/CLAUDE.md](src/services/CLAUDE.md) — background services, daemon thread pattern

## Coding Conventions
- All paths resolved in `config.py` — never hardcode absolute paths
- YOLO fallback chain: `.onnx` → `.pt` (auto in `config.py`)
- Encoding file always backed up before overwrite (in `encoder.py`)
- Services run as daemon threads — check `notification_worker.py` for pattern
