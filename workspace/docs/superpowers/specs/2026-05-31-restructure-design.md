# Thiết Kế Tái Cấu Trúc Codebase — Face Recognition Smart Home

**Ngày:** 2026-05-31  
**Phạm vi:** Medium restructuring — layer architecture  
**Mục tiêu:** Dễ maintain, dễ mở rộng, ranh giới rõ ràng giữa các layer

---

## 1. Vấn Đề Hiện Tại

| File | Dòng | Vấn đề |
|------|------|--------|
| `src/training_manager.py` | 1,232 | 4 class khác nhau trong 1 file |
| `src/core/frame_extractor.py` | 775 | 3 class trong 1 file |
| `src/recognizer.py` | 747 | Nằm sai layer (là core component nhưng ở root src/) |
| `src/benchmark_models.py`, `optimize_encodings.py`, `augment_dataset.py`, `export_onnx.py` | — | Scripts tiện ích lẫn lộn với application code |
| `src/` root | — | Không có ranh giới rõ ràng giữa business logic và core utilities |

---

## 2. Cấu Trúc Mới

```
workspace/
├── face_manager.py               # Entry point MỚI: cập nhật data, train lại model
│
├── scripts/                      # Maintenance tools — chạy độc lập
│   ├── benchmark_models.py
│   ├── optimize_encodings.py
│   ├── augment_dataset.py
│   └── export_onnx.py
│
├── src/
│   ├── core/                     # Low-level, reusable, không chứa business logic
│   │   ├── __init__.py
│   │   ├── recognizer.py         # FaceRecognizer (move từ src/recognizer.py)
│   │   ├── encoder.py            # rebuild_encodings, update_person_encodings
│   │   ├── face_utils.py         # extract_face_region
│   │   ├── motion_guard.py       # MotionGuard
│   │   ├── recognition_stabilizer.py  # RecognitionStabilizer
│   │   ├── frame_extractor.py    # VideoFrameExtractor + ExtractedFrame
│   │   ├── frame_quality.py      # FrameQualityChecker (tách từ frame_extractor.py)
│   │   └── frame_diversity.py    # FrameDiversityFilter (tách từ frame_extractor.py)
│   │
│   ├── training/                 # Training domain — quản lý người và dữ liệu
│   │   ├── __init__.py
│   │   ├── person_manager.py     # PersonManager — CRUD persons
│   │   ├── data_collection.py    # DataCollectionSession — thu thập từ camera/video
│   │   ├── extraction_settings.py# ExtractionSettings + presets
│   │   └── cli.py                # InteractiveCLI — terminal UI (questionary)
│   │
│   ├── services/                 # Smart home business logic
│   │   ├── __init__.py
│   │   ├── alert_manager.py      # Cảnh báo người lạ (Telegram, sound)
│   │   ├── device_dispatcher.py  # Điều khiển thiết bị (MQTT, Serial)
│   │   ├── event_logger.py       # Ghi lịch sử vào nhà (JSONL)
│   │   └── notification_worker.py# Background thread xử lý I/O
│   │
│   ├── config.py                 # Toàn bộ tham số hệ thống
│   └── main.py                   # Entry point hiện tại: nhận diện real-time (giữ nguyên vị trí)
│
├── tests/                        # Giữ nguyên, cập nhật imports
├── docs/
├── family_images/
├── model/
├── data/
└── logs/
```

---

## 3. Nguyên Tắc Phân Layer

### `src/core/` — Core layer
- **Không biết** business logic smart home (không import services/)
- **Không biết** training domain (không import training/)
- Có thể dùng lại ở project khác mà không cần sửa
- Chỉ phụ thuộc: `config.py`, thư viện bên ngoài (cv2, numpy, face_recognition, YOLO)

### `src/training/` — Training domain
- **Không biết** services/ (không import alert_manager, dispatcher...)
- Import từ `core/` (encoder, frame_extractor, face_utils)
- Entry point: `face_manager.py` ở root gọi vào `training/cli.py`

### `src/services/` — Services layer
- Business logic smart home: alert, dispatch, log, notify
- Import từ `core/` nhưng **không import** training/
- Được dùng bởi `main.py`

### `scripts/` — Standalone tools
- Import từ `src/core/` (một chiều)
- Không import services/, không import training/
- Mỗi script tự thêm `src/` vào `sys.path`

### Dependency graph (không có circular dependency)
```
face_manager.py  →  training/cli.py  →  training/{person_manager, data_collection, extraction_settings}
                                      →  core/{encoder, frame_extractor, frame_quality, frame_diversity}

main.py          →  core/recognizer.py
                 →  services/{alert_manager, device_dispatcher, event_logger, notification_worker}
                 →  core/recognition_stabilizer.py
                 →  config.py

scripts/*        →  core/*  (một chiều)
```

---

## 4. Chi Tiết Tách File Lớn

### 4.1 `frame_extractor.py` → 3 files

| Class | File mới |
|-------|----------|
| `FrameQualityChecker` | `src/core/frame_quality.py` |
| `FrameDiversityFilter` | `src/core/frame_diversity.py` |
| `VideoFrameExtractor` + `ExtractedFrame` | `src/core/frame_extractor.py` (giữ tên, bỏ 2 class trên) |

`frame_extractor.py` import từ `frame_quality.py` và `frame_diversity.py`.  
Code bên ngoài import từ `frame_extractor` như cũ — không breaking change.

### 4.2 `training_manager.py` → 4 files trong `src/training/`

| Class | File mới |
|-------|----------|
| `PersonManager` | `src/training/person_manager.py` |
| `DataCollectionSession` | `src/training/data_collection.py` |
| `ExtractionSettings` | `src/training/extraction_settings.py` |
| `InteractiveCLI` | `src/training/cli.py` |

Helper functions (`_float_validator`, `_int_validator`, `_help_*`) đặt trong file sử dụng chúng hoặc `src/training/cli.py`.

### 4.3 Entry points mới tại root

**`face_manager.py`** — thin wrapper:
```python
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from training.cli import InteractiveCLI

if __name__ == "__main__":
    InteractiveCLI().run()
```

**`src/main.py`** — cập nhật imports (giữ nguyên vị trí `src/`):
```python
from core.recognizer import FaceRecognizer
from services.alert_manager import AlertManager
from services.device_dispatcher import DeviceDispatcher
from services.event_logger import EventLogger
from services.notification_worker import NotificationWorker
from core.recognition_stabilizer import RecognitionStabilizer
```

Chạy như cũ: `python src/main.py`

---

## 5. Chiến Lược Migrate (thứ tự thực hiện)

Mỗi bước chạy `pytest` trước khi sang bước tiếp theo.

| Bước | Hành động | Rủi ro |
|------|-----------|--------|
| 1 | Tạo `src/services/`, `src/training/`, `scripts/` + `__init__.py` | Không |
| 2 | Move 4 services files vào `src/services/`, cập nhật `main.py` | Thấp |
| 3 | Tách `frame_extractor.py` → `frame_quality.py` + `frame_diversity.py` | Trung bình |
| 4 | Move `src/recognizer.py` → `src/core/recognizer.py`, cập nhật `main.py` | Thấp |
| 5 | Tách `training_manager.py` → 4 files trong `src/training/` | Cao (file lớn nhất) |
| 6 | Move 4 scripts ra `scripts/`, thêm `sys.path` | Thấp |
| 7 | Tạo `face_manager.py` ở root | Không |
| 8 | Chạy toàn bộ test suite, fix nếu có lỗi | — |

---

## 6. Quy Tắc Mở Rộng Về Sau

- Thêm tính năng smart home mới → tạo file mới trong `src/services/`
- Thêm core algorithm mới → tạo file mới trong `src/core/`
- Thêm maintenance tool mới → tạo file mới trong `scripts/`
- Không bao giờ để 1 file vượt quá ~400 dòng — đó là tín hiệu cần tách
- `config.py` là nguồn duy nhất của mọi tham số — không hardcode ở file khác
