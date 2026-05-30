---
name: run-workspace
description: Run, build, test, and smoke-test the Family Face Recognition System. Use when asked to run, start, test, verify, or confirm changes to the face recognition pipeline, training manager, augmentation, encoding, or recognition modules.
---

This is a Python library + CLI for family face recognition, combining YOLO (face detection) with dlib/face_recognition (identification). Core modules live in `src/`. The project has no web UI or desktop GUI — the interactive surfaces are a questionary CLI (`training_manager.py`) and a camera window (`recognizer.py`). **The agent path is the smoke driver and test suite**, not the interactive CLI.

Driver: `.claude/skills/run-workspace/driver.py`
All paths below are relative to the workspace root `/home/dkhai/workspace/`.

## Prerequisites

Python 3.13 is the active interpreter (`python3`). Install all deps fresh on a new machine:

```bash
# C++ compiler needed to build dlib from source
sudo apt-get update
sudo apt-get install -y g++-12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Python packages
pip3 install dlib face_recognition onnxruntime scikit-image questionary pytest
# ultralytics and cv2 (opencv-python) are typically pre-installed
```

Expected package check:
```bash
python3 -c "import face_recognition, ultralytics, cv2, onnxruntime, skimage, questionary; print('all ok')"
```

## Run — agent path (smoke driver)

```bash
cd /home/dkhai/workspace
python3 .claude/skills/run-workspace/driver.py
```

Exercises 6 checks without a camera, YOLO model inference, or touching real encodings:
- Module imports (`config`, `core.*`, `training_manager`, `augment_dataset`)
- `FrameQualityChecker` with synthetic images (sharpness / brightness / size gates)
- `FrameDiversityFilter` temporal deduplication
- `PersonManager` CRUD in a temp dataset with `rebuild=False`
- `augment_person()` with a synthetic face image in a temp dataset
- `rebuild_encodings()` on an empty temp dataset

Expected output:
```
  PASS  imports
  PASS  FrameQualityChecker
  PASS  FrameDiversityFilter
  PASS  PersonManager CRUD
  PASS  augment_person
  PASS  rebuild_encodings (empty dataset)

Results: 6 passed, 0 failed
All smoke checks passed.
```

## Run — test suite

```bash
cd /home/dkhai/workspace
python3 -m pytest tests/ -v
```

93 tests in 7 files covering alert manager, device dispatcher, event logger, frame extractor, frame quality, integration pipeline, person manager CRUD, and stabilizer.

## Direct invocation of library code

Most PRs touch internals. Call modules directly without launching the CLI:

```python
import sys; sys.path.insert(0, 'src')
import config
from core.frame_extractor import FrameQualityChecker, FrameDiversityFilter
from core.encoder import rebuild_encodings
from training_manager import PersonManager
```

## Run — human/interactive path (camera required)

```bash
cd /home/dkhai/workspace/src

# Collect training data (interactive questionary menu)
python3 training_manager.py

# Real-time recognition (opens camera window, press q to quit)
python3 recognizer.py --camera 0 --tolerance 0.6

# Augment dataset (no-encode to skip pkl rebuild)
python3 augment_dataset.py --person Khai --no-preview --no-encode

# Rebuild encodings from dataset
python3 -c "
import sys; sys.path.insert(0, 'src')
import config
from core.encoder import rebuild_encodings
rebuild_encodings(config.DATASET_PATH, config.MODEL_PATH,
    'model/face_encodings_hybrid.pkl')
"
```

These require a real camera or video file. They are not runnable headless.

## Gotchas

**`PersonManager._rebuild()` always writes to the real `config.ENCODINGS_DIR`** regardless of the `dataset_path` you pass. Using a temp dir as `dataset_path` will silently overwrite the production `model/face_encodings_hybrid.pkl` with 0 encodings. Always pass `rebuild=False` in tests and smoke scripts. The backup at `model/face_encodings_hybrid_backup.pkl` can restore it: `cp model/face_encodings_hybrid_backup.pkl model/face_encodings_hybrid.pkl`.

**Python version: use `python3`, not `python3.11`**. The system Python is now 3.13. All packages installed under `python3` / `pip3`.

**`pkg_resources` deprecation warning** from `face_recognition_models` is harmless. It appears on every import of `face_recognition` and does not affect functionality.

**`FrameDiversityFilter.is_diverse()` requires `min_frame_gap` frames between candidates**, not just SSIM difference. Two visually different frames at frame numbers 0 and 1 with default `min_frame_gap=10` will still both fail. Use `FrameDiversityFilter(min_frame_gap=1)` in tests.

**YOLO `.onnx` model not present by default**. `config.MODEL_PATH` falls back to `yolov11n-face.pt`. To create the ONNX version: `cd src && python3 export_onnx.py`. The `.pt` fallback works fine.

**dlib requires a C++ compiler to build from source**. On fresh Debian/Ubuntu, `g++` may not exist even after installing `g++-12` — run `sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100` to create the symlink.

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'face_recognition'` | `pip3 install face_recognition` (requires dlib; see Prerequisites) |
| `No CMAKE_CXX_COMPILER could be found` during dlib build | Install g++ and create symlink: `sudo apt-get install g++-12 && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100` |
| `ModuleNotFoundError: No module named 'onnxruntime'` | `pip3 install onnxruntime` |
| `ModuleNotFoundError: No module named 'skimage'` | `pip3 install scikit-image` |
| `ModuleNotFoundError: No module named 'questionary'` | `pip3 install questionary` |
| `No module named pytest` | `pip3 install pytest` |
| `face_encodings_hybrid.pkl` has 0 encodings | Restore: `cp model/face_encodings_hybrid_backup.pkl model/face_encodings_hybrid.pkl` |
| YOLO model not found | Check `src/yolo/yolov11n-face.pt` exists |
| Camera index error | `ls /dev/video*`; pass correct `--camera N` |
