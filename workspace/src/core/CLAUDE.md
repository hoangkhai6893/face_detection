# src/core — Recognition Pipeline

## Modules & Responsibilities

| File | Role |
|---|---|
| `recognizer.py` | `RecognitionPipeline` — orchestrates full detect→recognize loop |
| `encoder.py` | `rebuild_encodings()` — reads dataset, writes `model/encodings.pkl` |
| `frame_extractor.py` | Extract quality frames from video for training |
| `frame_quality.py` | Quality score: Laplacian variance + face size |
| `frame_diversity.py` | Filter near-duplicate frames before saving |
| `recognition_stabilizer.py` | Temporal smoothing — prevents name flickering across frames |
| `motion_guard.py` | `MotionGuard` — IDLE/ACTIVE state machine: tiết kiệm CPU khi không có người, probe burst khi timeout |
| `face_utils.py` | `extract_face_region()` — crop + clamp helper |

## Recognition Decision Logic

```
YOLO bbox → top-K nearest encodings (K=5)
          → weighted vote: weight = (1 - distance) per encoding
          → winner = person with highest total weight
          → if (winner_dist - runnerup_dist) < CONFUSION_MARGIN → Unknown
          → else → winner name
```

Key constants (all in `src/config.py`): `RECOGNITION_TOP_K=5`, `CONFUSION_MARGIN=0.10`, `TOLERANCE=0.5`.

Voting dùng **distance-weighted**: encoding gần hơn (nhỏ distance) có trọng số lớn hơn — phân biệt tốt hơn khi 2 thành viên có embedding gần nhau.

## Encoding Pipeline (encoder.py)

`_extract_face_encoding()`: YOLO detect → truyền bbox trực tiếp làm `known_face_locations` vào `face_encodings()` — **bỏ bước HOG trung gian**, nhất quán với `recognizer.py`.

`rebuild_encodings()`: sau khi encode từng ảnh, gọi `_cluster_to_max()` per person → giới hạn `MAX_ENCODINGS_PER_PERSON=50` ngay trong rebuild (không cần bước optimize riêng).

`update_person_encodings()`: tương tự, gộp encoding cũ + mới → cluster → lưu.

## Encoding File Format

```python
# model/encodings.pkl
{"encodings": List[np.ndarray],  # 128-d vectors
 "names":     List[str]}         # parallel list of person names
```

Always backed up before overwrite — see `encoder.py`.

## MotionGuard — State Machine

`motion_guard.py` chạy state machine IDLE/ACTIVE để tiết kiệm CPU khi không có người:

```
IDLE  → check motion mỗi 6 frame (absdiff + MOG2, 160×90px, ~0.6ms)
      → motion detected          → ACTIVE (trong 0.2s)
      → timeout 60s              → probe burst 5 frame (YOLO only, không đổi state)
           → thấy mặt            → ACTIVE
           → không thấy mặt     → reset timer, vẫn IDLE

ACTIVE → full pipeline mỗi frame (YOLO + dlib)
       → 150 frame không thấy mặt → IDLE
```

**API:**
- `should_process(frame)` → `bool` — gọi mỗi frame, trả True khi nên chạy YOLO
- `report_faces(count)` — gọi sau YOLO, quyết định đổi state

**Quan trọng:** probe burst KHÔNG đổi state sang ACTIVE — chỉ `report_faces()` mới đổi state khi thấy mặt. Điều này loại bỏ vòng lặp ACTIVE↔IDLE vô tận khi không có người ở nhà.
