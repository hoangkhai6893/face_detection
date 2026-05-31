# src/training — Data Collection & Management

## Modules

| File | Role |
|---|---|
| `cli.py` | CLI entry — menu / argument parsing for training commands |
| `data_collection.py` | Capture frames from camera or video file |
| `person_manager.py` | CRUD: create / list / delete / reset persons in dataset |
| `extraction_settings.py` | Per-run config (overrides config.py defaults for a session) |

## Dataset Layout

```
family_images/
  <person_name>/
    001.jpg
    002.jpg
    ...          ← max MAX_SAMPLES_PER_PERSON (50) raw images
```

After `rebuild_encodings()` → `model/encodings.pkl` (max `MAX_ENCODINGS_PER_PERSON=50` after clustering per person).

## Collection Pipeline

```
camera/video → FrameExtractor
             → quality filter (Laplacian ≥ 15, face ≥ 70px)
             → diversity filter (skip near-duplicates)
             → save to family_images/<person>/
```

Thresholds for video are lower than live camera — compressed video degrades Laplacian significantly.
