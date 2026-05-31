# Phase 4 — Benchmark Trên Pi 5 & Tune Accuracy

> **Mục tiêu:** Đo thực tế trên Pi5 → chốt tham số tối ưu → xác nhận hệ thống đạt yêu cầu.
> **Nguyên tắc:** KHÔNG đoán số. Mọi quyết định config dựa trên số đo thật.

---

## 4.1 Benchmark Performance

### Script đo thời gian từng thành phần

```bash
# scripts/benchmark_pipeline.py
# Usage: python scripts/benchmark_pipeline.py --image <path> --n 100
```

```python
"""
Đo thời gian từng bước trong pipeline recognition.
Chạy N lần lấy trung bình, đo trên Pi5 thật.
"""
import time, cv2, numpy as np, argparse

def benchmark(image_path: str, n: int = 100):
    import config
    from core.detector import YuNetDetector
    from core.embedder import SFaceEmbedder

    frame = cv2.imread(image_path)
    assert frame is not None, f"Không đọc được: {image_path}"

    det = YuNetDetector(config.YUNET_MODEL_PATH)
    emb = SFaceEmbedder(config.SFACE_MODEL_PATH)

    # Warmup
    for _ in range(5):
        dets = det.detect(frame)

    # Benchmark detection
    t0 = time.time()
    for _ in range(n):
        dets = det.detect(frame)
    det_ms = (time.time() - t0) / n * 1000

    # Benchmark encode (nếu có face)
    enc_ms = None
    if dets:
        t0 = time.time()
        for _ in range(n):
            emb.encode(frame, dets[0])
        enc_ms = (time.time() - t0) / n * 1000

    # Benchmark distance matching (vs 300 encodings)
    dist_ms = None
    if dets:
        enc = emb.encode(frame, dets[0])
        known = [np.random.rand(128).astype(np.float32) for _ in range(300)]
        t0 = time.time()
        for _ in range(n):
            emb.batch_distance(known, enc)
        dist_ms = (time.time() - t0) / n * 1000

    print(f"\n=== Benchmark (n={n}) ===")
    print(f"YuNet detect:     {det_ms:.1f}ms")
    print(f"SFace encode:     {enc_ms:.1f}ms" if enc_ms else "SFace encode: no face detected")
    print(f"Distance (×300):  {dist_ms:.2f}ms" if dist_ms else "Distance: skipped")
    if enc_ms and dist_ms:
        total = det_ms + enc_ms + dist_ms
        print(f"Total per face:   {total:.1f}ms  (~{1000/total:.1f} fps theoretical)")
    print(f"\n[IDLE] MOG2 on 160×90: ~0.6ms (unchanged)")
    print(f"[ACTIVE] YOLO+encode+distance per frame: ~{det_ms + (enc_ms or 0) + (dist_ms or 0):.0f}ms")
```

### Các metric cần ghi lại trên Pi5:

| Metric | Đo bằng cách | Target |
|---|---|---|
| `detect_ms` | benchmark_pipeline.py | < 30ms |
| `encode_ms` | benchmark_pipeline.py | < 50ms |
| `distance_ms` | benchmark_pipeline.py | < 5ms |
| `total_per_face_ms` | tổng 3 trên | < 80ms |
| `fps_active_1face` | live run + FPS counter | > 10fps |
| `cpu_active` | `htop` hoặc `pidstat` | < 60% |
| `cpu_idle` | live run khi nhà trống | < 5% |
| `temp_active` | `vcgencmd measure_temp` | < 70°C |
| `ram_mb` | `ps aux` | < 350MB |

### Benchmark end-to-end (live, 10 phút)

```bash
# Trên Pi5: chạy 10 phút với 1 người đứng trước camera
python src/main.py --headless --active-process-every 2 -v 2>&1 | tee /tmp/bench_10min.log

# Sau đó phân tích:
grep "FPS" /tmp/bench_10min.log | awk '{sum+=$2; n++} END {print "avg FPS:", sum/n}'
grep "HOG fallback" /tmp/bench_10min.log | wc -l   # số lần HOG trigger (kỳ vọng: 0)
grep "HEALTH" /tmp/bench_10min.log                  # health log mỗi giờ

# Nhiệt độ
watch -n 5 vcgencmd measure_temp   # theo dõi trong 10 phút
```

---

## 4.2 Tune Thresholds Cosine (SFace)

### Context: tại sao cần tune

SFace official threshold: cosine_score ≥ 0.363 → same person.
Nhưng đó là threshold cho dataset chung (LFW). Gia đình indoor có thể:
- Gần hơn (cùng nhà, ánh sáng tốt, phần mặt lớn) → threshold cần điều chỉnh
- Người thân giống nhau (bố/con) → margin cần đủ lớn để phân biệt

### Quy trình đo threshold tối ưu

**Bước 1: Thu thập cosine scores từ test data**

```bash
python scripts/collect_scores.py \
  --test-dir test_data/ \
  --encodings model/face_encodings_hybrid.pkl \
  --output scores.csv
```

```python
# scripts/collect_scores.py
# Output scores.csv: image_path, true_label, predicted_name, cosine_distance, correct (bool)
```

**Bước 2: Vẽ distribution và tìm điểm tối ưu**

```python
# scripts/tune_threshold.py
import pandas as pd, numpy as np

df = pd.read_csv("scores.csv")

# Phân tách same-person và diff-person
same = df[df["true_label"] == df["predicted_name"]]["cosine_distance"]
diff = df[df["true_label"] != df["predicted_name"]]["cosine_distance"]

print(f"Same person distances: mean={same.mean():.3f}, p95={same.quantile(0.95):.3f}")
print(f"Diff person distances: mean={diff.mean():.3f}, p5={diff.quantile(0.05):.3f}")

# Tìm threshold tối ưu: maximize (true_positive - false_positive)
for t in np.arange(0.4, 0.8, 0.01):
    tp = (same < t).mean()   # % same-person được nhận đúng
    fp = (diff < t).mean()   # % diff-person bị nhận nhầm
    print(f"T={t:.2f}: TPR={tp:.2%}  FPR={fp:.2%}  F1-like={2*tp*(1-fp)/(tp+(1-fp)):.3f}")
```

**Bước 3: Chọn threshold theo ưu tiên**

Hai mục tiêu đôi khi đối lập — chọn theo ưu tiên của bạn:

| Ưu tiên | Threshold thấp (strict) | Threshold cao (lenient) |
|---|---|---|
| Không nhầm người nhà này sang người nhà khác | ✅ tốt | ⚠️ rủi ro |
| Ít "Unknown" giả với người nhà | ⚠️ nhiều | ✅ ít |
| Bắt người lạ | ✅ bắt nhiều | ⚠️ bỏ sót |

**Khuyến nghị:** ưu tiên `confusion_rate = 0` (không bao giờ nhầm người nhà).
Nhận "Unknown" giả đôi khi → chỉ là không bật đèn, không báo nhầm.

### Config sau khi tune

```bash
# .env trên Pi5
TOLERANCE_SFACE=0.XX          # thay bằng giá trị đo được
CONFUSION_MARGIN_SFACE=0.XX
```

---

## 4.3 Tune MotionGuard Cho Điều Kiện Thực

### Điều chỉnh nếu quá nhạy với ánh sáng (nhiều false ACTIVE)

```bash
# Tăng MOG2 threshold (% pixels foreground để confirm motion)
MOTION_MOG2_THRESHOLD=8   # default 5 → tăng lên 8 nếu đèn bật/tắt trigger nhầm

# Tăng absdiff threshold
MOTION_ABSDIFF_THRESHOLD=12   # default 8
```

### Điều chỉnh nếu người đứng yên không được detect (missed ACTIVE)

```bash
# Giảm probe timeout (probe thường xuyên hơn)
MOTION_MAX_IDLE_SEC=30   # default 60

# Tăng probe burst (nhiều frame hơn mỗi lần probe)
IDLE_PROBE_BURST=8   # default 5
```

### Điều chỉnh thời gian về IDLE sau khi người rời khỏi

```bash
# IDLE_NO_FACE_FRAMES frames × (1/fps) giây
# Default 150 frames ÷ 15fps ≈ 10s sau khi không thấy mặt → IDLE
# Nếu muốn về IDLE nhanh hơn:
IDLE_NO_FACE_FRAMES=60   # ≈ 4s
```

---

## 4.4 Tune Camera

### Nếu face quá nhỏ (camera xa)

```bash
YOLO_INPUT_WIDTH=416   # giữ nguyên (tốt hơn 320 cho khuôn mặt nhỏ)
# Hoặc:
FRAME_WIDTH=1280       # capture resolution cao hơn rồi scale
```

### Nếu FPS vẫn thấp hơn mong đợi

```bash
ACTIVE_PROCESS_EVERY=3   # skip thêm — giảm xuống 1/3 pipeline load
YOLO_INPUT_WIDTH=320     # YuNet nhỏ hơn (không ảnh hưởng nhiều accuracy indoor)
```

---

## 4.5 Kết Quả Benchmark Mẫu (điền sau khi đo)

```
=== Pi 5 4GB Benchmark — YYYY-MM-DD ===
Model: YuNet INT8 + SFace INT8

YuNet detect:    ___ms
SFace encode:    ___ms
Distance ×300:   ___ms
Total per face:  ___ms → ___fps theoretical

Live (10 phút, 1 mặt):
  FPS avg:         ___
  FPS min:         ___
  CPU active:      ___%
  CPU idle:        ___%
  Temp active:     ___°C
  Temp idle:       ___°C
  RAM:             ___MB
  HOG fallbacks:   0   (kỳ vọng)

Accuracy (validation gate):
  Known accuracy:    ___%  (Phase 2: ___%)
  False-unknown:     ___%  (Phase 2: ___%)
  Confusion:         ___%  (Phase 2: ___%)

Decision: PASS / FAIL / NEEDS_TUNING
Final TOLERANCE_SFACE: 0.XX
Final CONFUSION_MARGIN_SFACE: 0.XX
```

---

## 4.6 Sign-off Checklist

Hệ thống sẵn sàng deploy production khi tất cả đều ✅:

```
[ ] detect_ms < 30ms trên Pi5
[ ] encode_ms < 50ms trên Pi5
[ ] fps_active >= 10fps với 1 mặt
[ ] cpu_active < 60%
[ ] temp_active < 70°C sau 30 phút liên tục
[ ] cpu_idle < 5% (nhà trống)
[ ] confusion_rate = 0% (không nhầm người nhà)
[ ] known_accuracy >= 90%
[ ] HOG fallback = 0 (trong điều kiện ánh sáng bình thường)
[ ] systemd service tự restart sau khi kill/camera rút
[ ] Log rotation hoạt động (file mỗi ngày, xóa file > 30 ngày)
[ ] pytest tests/ -v → tất cả pass
[ ] Không import dlib / face_recognition / ultralytics trong src/ (đã gỡ)
```
