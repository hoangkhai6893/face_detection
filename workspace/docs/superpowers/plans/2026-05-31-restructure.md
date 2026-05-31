# Restructure Codebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái cấu trúc codebase từ flat layout sang layer architecture (core / training / services) để dễ maintain và mở rộng.

**Architecture:** Ba layer rõ ràng — `src/core/` (reusable, không có business logic), `src/services/` (smart home logic), `src/training/` (training domain). Scripts tiện ích tách ra `scripts/`. Entry points: `face_manager.py` (train) và `src/main.py` (run).

**Tech Stack:** Python 3.11+, pytest, cv2, face_recognition, ultralytics YOLO, questionary

---

## File Map

### Tạo mới
- `src/services/__init__.py`
- `src/training/__init__.py`
- `scripts/` (thư mục)
- `src/core/frame_quality.py` — FrameQualityChecker
- `src/core/frame_diversity.py` — FrameDiversityFilter
- `src/training/extraction_settings.py` — ExtractionSettings + presets
- `src/training/person_manager.py` — PersonInfo + PersonManager
- `src/training/data_collection.py` — CollectionResult + DataCollectionSession
- `src/training/cli.py` — InteractiveCLI + helpers + main()
- `face_manager.py` (root workspace)

### Di chuyển (nội dung giữ nguyên, chỉ cập nhật imports)
- `src/alert_manager.py` → `src/services/alert_manager.py`
- `src/device_dispatcher.py` → `src/services/device_dispatcher.py`
- `src/event_logger.py` → `src/services/event_logger.py`
- `src/notification_worker.py` → `src/services/notification_worker.py`
- `src/recognizer.py` → `src/core/recognizer.py`
- `src/benchmark_models.py` → `scripts/benchmark_models.py`
- `src/optimize_encodings.py` → `scripts/optimize_encodings.py`
- `src/augment_dataset.py` → `scripts/augment_dataset.py`
- `src/export_onnx.py` → `scripts/export_onnx.py`

### Sửa imports
- `src/main.py` — cập nhật 4 service imports + recognizer import
- `src/core/frame_extractor.py` — import từ frame_quality + frame_diversity thay vì định nghĩa inline
- `tests/test_alert_manager.py`
- `tests/test_device_dispatcher.py`
- `tests/test_event_logger.py`
- `tests/test_frame_quality.py`
- `tests/test_frame_extractor.py`
- `tests/test_integration.py`
- `tests/test_person_manager.py`

### Xoá
- `src/training_manager.py` (sau khi tách xong Task 5)
- `src/recognizer.py` (sau khi move xong Task 4)
- `src/alert_manager.py`, `src/device_dispatcher.py`, `src/event_logger.py`, `src/notification_worker.py` (sau Task 2)
- `src/benchmark_models.py`, `src/optimize_encodings.py`, `src/augment_dataset.py`, `src/export_onnx.py` (sau Task 6)

---

## Task 1: Baseline + Tạo thư mục

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/training/__init__.py`
- Create: `scripts/` (empty dir)

- [ ] **Step 1: Chạy toàn bộ test suite để xác nhận baseline**

```bash
cd /home/ubuntu/workspace
python -m pytest tests/ -v 2>&1 | tail -20
```

Ghi lại số test pass. Nếu có test fail ngay từ đầu, fix trước khi tiếp tục.

- [ ] **Step 2: Tạo thư mục và `__init__.py`**

```bash
mkdir -p /home/ubuntu/workspace/src/services
mkdir -p /home/ubuntu/workspace/src/training
mkdir -p /home/ubuntu/workspace/scripts
touch /home/ubuntu/workspace/src/services/__init__.py
touch /home/ubuntu/workspace/src/training/__init__.py
```

- [ ] **Step 3: Chạy lại tests — phải giống baseline**

```bash
python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: cùng số pass như bước 1.

- [ ] **Step 4: Commit**

```bash
git add src/services/__init__.py src/training/__init__.py
git commit -m "chore: create services/, training/, scripts/ directories"
```

---

## Task 2: Move services vào `src/services/`

**Files:**
- Move: `src/alert_manager.py` → `src/services/alert_manager.py`
- Move: `src/device_dispatcher.py` → `src/services/device_dispatcher.py`
- Move: `src/event_logger.py` → `src/services/event_logger.py`
- Move: `src/notification_worker.py` → `src/services/notification_worker.py`
- Modify: `src/main.py`
- Modify: `tests/test_alert_manager.py`
- Modify: `tests/test_device_dispatcher.py`
- Modify: `tests/test_event_logger.py`

- [ ] **Step 1: Move 4 service files**

```bash
mv /home/ubuntu/workspace/src/alert_manager.py /home/ubuntu/workspace/src/services/
mv /home/ubuntu/workspace/src/device_dispatcher.py /home/ubuntu/workspace/src/services/
mv /home/ubuntu/workspace/src/event_logger.py /home/ubuntu/workspace/src/services/
mv /home/ubuntu/workspace/src/notification_worker.py /home/ubuntu/workspace/src/services/
```

- [ ] **Step 2: Cập nhật imports trong `src/main.py`**

Tìm và thay thế các dòng import sau trong `src/main.py`:

```python
# Trước
from alert_manager import AlertManager
from event_logger import EventLogger
from device_dispatcher import DeviceDispatcher
from notification_worker import NotificationWorker

# Sau
from services.alert_manager import AlertManager
from services.event_logger import EventLogger
from services.device_dispatcher import DeviceDispatcher
from services.notification_worker import NotificationWorker
```

- [ ] **Step 3: Cập nhật import trong `tests/test_alert_manager.py`**

```python
# Trước (dòng 9)
from alert_manager import AlertManager

# Sau
from services.alert_manager import AlertManager
```

- [ ] **Step 4: Cập nhật import trong `tests/test_device_dispatcher.py`**

```python
# Trước (dòng 8-12)
from device_dispatcher import (
    DeviceDispatcher, PersonProfile, DeviceAction,
    LogBackend, WebhookBackend, MqttBackend, SerialBackend,
)

# Sau
from services.device_dispatcher import (
    DeviceDispatcher, PersonProfile, DeviceAction,
    LogBackend, WebhookBackend, MqttBackend, SerialBackend,
)
```

- [ ] **Step 5: Cập nhật import trong `tests/test_event_logger.py`**

```python
# Trước (dòng 8)
from event_logger import EventLogger

# Sau
from services.event_logger import EventLogger
```

- [ ] **Step 6: Chạy tests**

```bash
python -m pytest tests/test_alert_manager.py tests/test_device_dispatcher.py tests/test_event_logger.py -v
```

Expected: tất cả PASS.

- [ ] **Step 7: Commit**

```bash
git add src/services/ src/main.py tests/test_alert_manager.py tests/test_device_dispatcher.py tests/test_event_logger.py
git commit -m "refactor: move services to src/services/"
```

---

## Task 3: Tách `frame_extractor.py` → `frame_quality.py` + `frame_diversity.py`

**Files:**
- Create: `src/core/frame_quality.py`
- Create: `src/core/frame_diversity.py`
- Modify: `src/core/frame_extractor.py`
- Modify: `tests/test_frame_quality.py`
- Modify: `tests/test_frame_extractor.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Tạo `src/core/frame_quality.py`**

Copy `FrameQualityChecker` (dòng 55–188 trong `frame_extractor.py` hiện tại) sang file mới với header sau:

```python
#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config


# [Paste toàn bộ class FrameQualityChecker từ frame_extractor.py vào đây]
```

- [ ] **Step 2: Tạo `src/core/frame_diversity.py`**

Copy `FrameDiversityFilter` (dòng 190–237 trong `frame_extractor.py` hiện tại) sang file mới:

```python
#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# [Paste toàn bộ class FrameDiversityFilter từ frame_extractor.py vào đây]
```

- [ ] **Step 3: Cập nhật `src/core/frame_extractor.py`**

Xoá định nghĩa `FrameQualityChecker` và `FrameDiversityFilter` khỏi file. Thay bằng imports ở đầu file (sau các imports hiện có):

```python
from core.frame_quality import FrameQualityChecker
from core.frame_diversity import FrameDiversityFilter
```

Giữ lại `ExtractedFrame` dataclass và `VideoFrameExtractor` trong file này.

- [ ] **Step 4: Cập nhật `tests/test_frame_quality.py`**

```python
# Trước (dòng 8)
from core.frame_extractor import FrameQualityChecker

# Sau
from core.frame_quality import FrameQualityChecker
```

- [ ] **Step 5: Cập nhật `tests/test_frame_extractor.py`**

```python
# Trước (dòng 13-18)
from core.frame_extractor import (
    ExtractedFrame,
    FrameDiversityFilter,
    FrameQualityChecker,
    VideoFrameExtractor,
)

# Sau
from core.frame_extractor import ExtractedFrame, VideoFrameExtractor
from core.frame_quality import FrameQualityChecker
from core.frame_diversity import FrameDiversityFilter
```

- [ ] **Step 6: Cập nhật `tests/test_integration.py`**

```python
# Trước (dòng 14-19)
from core.frame_extractor import (
    FrameDiversityFilter,
    FrameQualityChecker,
    VideoFrameExtractor,
)

# Sau
from core.frame_extractor import VideoFrameExtractor
from core.frame_quality import FrameQualityChecker
from core.frame_diversity import FrameDiversityFilter
```

- [ ] **Step 7: Chạy tests**

```bash
python -m pytest tests/test_frame_quality.py tests/test_frame_extractor.py tests/test_integration.py -v
```

Expected: tất cả PASS.

- [ ] **Step 8: Commit**

```bash
git add src/core/frame_quality.py src/core/frame_diversity.py src/core/frame_extractor.py \
        tests/test_frame_quality.py tests/test_frame_extractor.py tests/test_integration.py
git commit -m "refactor: split frame_extractor into frame_quality + frame_diversity"
```

---

## Task 4: Move `recognizer.py` vào `src/core/`

**Files:**
- Move: `src/recognizer.py` → `src/core/recognizer.py`
- Modify: `src/main.py`

- [ ] **Step 1: Move file**

```bash
mv /home/ubuntu/workspace/src/recognizer.py /home/ubuntu/workspace/src/core/recognizer.py
```

- [ ] **Step 2: Cập nhật import trong `src/main.py`**

```python
# Trước
from recognizer import FaceRecognizer, setup_logging

# Sau
from core.recognizer import FaceRecognizer, setup_logging
```

- [ ] **Step 3: Chạy toàn bộ tests**

```bash
python -m pytest tests/ -v 2>&1 | tail -15
```

Expected: tất cả PASS (không có test trực tiếp cho recognizer, nhưng integration test sẽ bắt lỗi import).

- [ ] **Step 4: Commit**

```bash
git add src/core/recognizer.py src/main.py
git commit -m "refactor: move FaceRecognizer to src/core/"
```

---

## Task 5: Tách `training_manager.py` thành 4 files

**Files:**
- Create: `src/training/extraction_settings.py`
- Create: `src/training/person_manager.py`
- Create: `src/training/data_collection.py`
- Create: `src/training/cli.py`
- Delete: `src/training_manager.py`
- Modify: `tests/test_person_manager.py`
- Modify: `tests/test_integration.py`

### Sub-task 5a: `extraction_settings.py`

- [ ] **Step 1: Tạo `src/training/extraction_settings.py`**

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import config

_SETTINGS_FILE = Path(config.DATASET_PATH).parent / "extraction_settings.json"


@dataclass
class ExtractionSettings:
    """All tunable parameters for frame extraction strictness.

    Persisted to *_SETTINGS_FILE* so the user does not have to re-tune every
    session.
    """
    min_laplacian: float = config.FRAME_MIN_LAPLACIAN
    min_face_px: int = config.FRAME_MIN_FACE_PX
    detect_conf: float = config.FRAME_DETECT_CONF
    ssim_threshold: float = 0.85
    min_frame_gap: int = 10
    frame_skip: int = 3

    def save(self, path: Path = _SETTINGS_FILE) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "min_laplacian": self.min_laplacian,
                "min_face_px": self.min_face_px,
                "detect_conf": self.detect_conf,
                "ssim_threshold": self.ssim_threshold,
                "min_frame_gap": self.min_frame_gap,
                "frame_skip": self.frame_skip,
            }, fh, indent=2)

    @classmethod
    def load(cls, path: Path = _SETTINGS_FILE) -> "ExtractionSettings":
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            defaults = cls()
            return cls(
                min_laplacian=float(data.get("min_laplacian", defaults.min_laplacian)),
                min_face_px=int(data.get("min_face_px", defaults.min_face_px)),
                detect_conf=float(data.get("detect_conf", defaults.detect_conf)),
                ssim_threshold=float(data.get("ssim_threshold", defaults.ssim_threshold)),
                min_frame_gap=int(data.get("min_frame_gap", defaults.min_frame_gap)),
                frame_skip=int(data.get("frame_skip", defaults.frame_skip)),
            )
        except Exception:
            return cls()


EXTRACTION_PRESETS: Dict[str, ExtractionSettings] = {
    "Thoai mai — bat nhieu frame, it reject nhat": ExtractionSettings(
        min_laplacian=5, min_face_px=50, detect_conf=0.08,
        ssim_threshold=0.93, min_frame_gap=5, frame_skip=2,
    ),
    "Binh thuong — khuyen nghi (mac dinh)": ExtractionSettings(),
    "Khat khe — chi lay frame chat luong cao": ExtractionSettings(
        min_laplacian=40, min_face_px=120, detect_conf=0.30,
        ssim_threshold=0.78, min_frame_gap=20, frame_skip=6,
    ),
}
```

Lưu ý: tên biến đổi từ `_EXTRACTION_PRESETS` (private) sang `EXTRACTION_PRESETS` (public) — cập nhật reference trong `cli.py` ở bước sau.

### Sub-task 5b: `person_manager.py`

- [ ] **Step 2: Tạo `src/training/person_manager.py`**

Copy `PersonInfo` dataclass (dòng 66–74) và `PersonManager` class (dòng 178–297) từ `training_manager.py`. Header:

```python
#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from core.encoder import rebuild_encodings

logger = logging.getLogger(__name__)

_INVALID_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]|^\.|^\.\.')
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class PersonInfo:
    name: str
    image_count: int
    augmented_count: int
    folder_path: Path
    created_date: str


# [Paste class PersonManager từ training_manager.py vào đây]
```

### Sub-task 5c: `data_collection.py`

- [ ] **Step 3: Tạo `src/training/data_collection.py`**

Copy `CollectionResult` dataclass (dòng 75–84) và `DataCollectionSession` class (dòng 298–481) từ `training_manager.py`. Header:

```python
#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config
from core.encoder import rebuild_encodings, update_person_encodings
from core.face_utils import extract_face_region
from core.frame_extractor import VideoFrameExtractor
from core.frame_quality import FrameQualityChecker
from core.frame_diversity import FrameDiversityFilter
from training.extraction_settings import ExtractionSettings

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    person_name: str
    frames_processed: int
    frames_saved: int
    frames_rejected: int
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    avg_quality: float = 0.0


# [Paste class DataCollectionSession từ training_manager.py vào đây]
```

### Sub-task 5d: `cli.py`

- [ ] **Step 4: Tạo `src/training/cli.py`**

Copy `InteractiveCLI` class (dòng 483–1222) và toàn bộ `_help_*` functions + `main()` từ `training_manager.py`. Header:

```python
#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Optional

import questionary

import config
from training.extraction_settings import ExtractionSettings, EXTRACTION_PRESETS
from training.person_manager import PersonManager
from training.data_collection import DataCollectionSession, CollectionResult

logger = logging.getLogger(__name__)


def _float_validator(lo: float, hi: float):
    # [Paste từ training_manager.py]

def _int_validator(lo: int, hi: int):
    # [Paste từ training_manager.py]


# [Paste class InteractiveCLI]
# [Paste _help_overview, _help_collect, _help_augment, _help_rebuild, _help_optimize, _help_settings]
# [Paste main()]
```

Cập nhật 2 lazy imports bên trong `InteractiveCLI._augment_flow()` và `InteractiveCLI` optimize flow để trỏ vào `scripts/`:

```python
# Trước (trong _augment_flow)
from augment_dataset import augment_person, AUGMENTATIONS

# Sau
_WORKSPACE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_WORKSPACE / "scripts"))
from augment_dataset import augment_person, AUGMENTATIONS
```

```python
# Trước (trong optimize flow)
from optimize_encodings import FaceEncodingOptimizer

# Sau
_WORKSPACE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_WORKSPACE / "scripts"))
from optimize_encodings import FaceEncodingOptimizer
```

### Sub-task 5e: Cập nhật tests + xoá file cũ

- [ ] **Step 5: Cập nhật `tests/test_person_manager.py`**

```python
# Trước (dòng 8)
from training_manager import PersonManager

# Sau
from training.person_manager import PersonManager
```

- [ ] **Step 6: Cập nhật `tests/test_integration.py`**

```python
# Trước (dòng 20-24)
from training_manager import (
    CollectionResult,
    DataCollectionSession,
    PersonManager,
)

# Sau
from training.person_manager import PersonManager
from training.data_collection import DataCollectionSession, CollectionResult
```

- [ ] **Step 7: Chạy tests liên quan**

```bash
python -m pytest tests/test_person_manager.py tests/test_integration.py -v
```

Expected: tất cả PASS.

- [ ] **Step 8: Chạy toàn bộ test suite**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: tất cả PASS.

- [ ] **Step 9: Xoá `src/training_manager.py`**

```bash
rm /home/ubuntu/workspace/src/training_manager.py
```

- [ ] **Step 10: Chạy lại tests để xác nhận không còn dependency**

```bash
python -m pytest tests/ -v 2>&1 | tail -10
```

- [ ] **Step 11: Commit**

```bash
git add src/training/ tests/test_person_manager.py tests/test_integration.py
git rm src/training_manager.py
git commit -m "refactor: split training_manager.py into src/training/"
```

---

## Task 6: Move scripts ra `scripts/`

**Files:**
- Move: `src/benchmark_models.py` → `scripts/benchmark_models.py`
- Move: `src/optimize_encodings.py` → `scripts/optimize_encodings.py`
- Move: `src/augment_dataset.py` → `scripts/augment_dataset.py`
- Move: `src/export_onnx.py` → `scripts/export_onnx.py`

- [ ] **Step 1: Move files**

```bash
mv /home/ubuntu/workspace/src/benchmark_models.py /home/ubuntu/workspace/scripts/
mv /home/ubuntu/workspace/src/optimize_encodings.py /home/ubuntu/workspace/scripts/
mv /home/ubuntu/workspace/src/augment_dataset.py /home/ubuntu/workspace/scripts/
mv /home/ubuntu/workspace/src/export_onnx.py /home/ubuntu/workspace/scripts/
```

- [ ] **Step 2: Thêm `sys.path` vào đầu mỗi script**

Mỗi file trong `scripts/` cần thêm các dòng sau ngay sau phần docstring và trước các imports khác:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

Thực hiện cho cả 4 files: `benchmark_models.py`, `optimize_encodings.py`, `augment_dataset.py`, `export_onnx.py`.

- [ ] **Step 3: Kiểm tra smoke test — mỗi script chạy được**

```bash
cd /home/ubuntu/workspace
python scripts/export_onnx.py --help 2>&1 | head -5
python scripts/benchmark_models.py --help 2>&1 | head -5
python scripts/augment_dataset.py --help 2>&1 | head -5
python scripts/optimize_encodings.py --help 2>&1 | head -5
```

Expected: không có `ModuleNotFoundError`.

- [ ] **Step 4: Chạy toàn bộ tests**

```bash
python -m pytest tests/ -v 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add scripts/
git rm src/benchmark_models.py src/optimize_encodings.py src/augment_dataset.py src/export_onnx.py
git commit -m "refactor: move standalone scripts to scripts/"
```

---

## Task 7: Tạo `face_manager.py` ở root

**Files:**
- Create: `face_manager.py` (root workspace)

- [ ] **Step 1: Tạo `face_manager.py`**

```python
#!/usr/bin/env python3
"""
face_manager.py — Cập nhật dữ liệu khuôn mặt và train lại model.

Chạy: python face_manager.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from training.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Kiểm tra chạy được (không crash khi import)**

```bash
cd /home/ubuntu/workspace
python -c "import sys; sys.path.insert(0, 'src'); from training.cli import InteractiveCLI; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Chạy toàn bộ test suite lần cuối**

```bash
python -m pytest tests/ -v
```

Expected: tất cả PASS — cùng số lượng với baseline ở Task 1.

- [ ] **Step 4: Commit**

```bash
git add face_manager.py
git commit -m "feat: add face_manager.py entry point for training/data management"
```

---

## Task 8: Verification

- [ ] **Step 1: Xác nhận cấu trúc thư mục**

```bash
find /home/ubuntu/workspace/src -name "*.py" | sort
find /home/ubuntu/workspace/scripts -name "*.py" | sort
```

Expected output:
```
src/config.py
src/core/__init__.py
src/core/encoder.py
src/core/face_utils.py
src/core/frame_diversity.py
src/core/frame_extractor.py
src/core/frame_quality.py
src/core/motion_guard.py
src/core/recognizer.py
src/core/recognition_stabilizer.py
src/main.py
src/services/__init__.py
src/services/alert_manager.py
src/services/device_dispatcher.py
src/services/event_logger.py
src/services/notification_worker.py
src/training/__init__.py
src/training/cli.py
src/training/data_collection.py
src/training/extraction_settings.py
src/training/person_manager.py

scripts/augment_dataset.py
scripts/benchmark_models.py
scripts/export_onnx.py
scripts/optimize_encodings.py
```

- [ ] **Step 2: Xác nhận không còn file cũ**

```bash
ls /home/ubuntu/workspace/src/recognizer.py 2>/dev/null && echo "FAIL: còn file cũ" || echo "OK"
ls /home/ubuntu/workspace/src/training_manager.py 2>/dev/null && echo "FAIL: còn file cũ" || echo "OK"
ls /home/ubuntu/workspace/src/alert_manager.py 2>/dev/null && echo "FAIL: còn file cũ" || echo "OK"
```

- [ ] **Step 3: Chạy toàn bộ tests lần cuối**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: tất cả PASS.

- [ ] **Step 4: Test 2 entry points chạy được**

```bash
# Test face_manager import
python -c "import sys; sys.path.insert(0,'src'); from training.cli import main; print('face_manager: OK')"

# Test main.py import
python -c "import sys; sys.path.insert(0,'src'); import config; from core.recognizer import FaceRecognizer; from services.alert_manager import AlertManager; print('main: OK')"
```

Expected: cả 2 dòng in `OK`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final restructure verification — all layers clean"
```
