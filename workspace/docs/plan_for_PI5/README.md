# Tài Liệu Kế Hoạch Nâng Cấp — Family Face Recognition

> **Mục tiêu:** Pi 5, headless, 24/7, stack OpenCV-native (YuNet + SFace)
> **Ngày tạo:** 2026-06-01

---

## Thứ Tự Đọc & Thực Hiện

| # | Tài liệu | Nội dung | Làm khi nào |
|---|---|---|---|
| 1 | [00_ARCHITECTURE_OVERVIEW.md](00_ARCHITECTURE_OVERVIEW.md) | Bức tranh toàn cảnh: stack cũ → mới, interface design, component map | Đọc trước để hiểu toàn bộ |
| 2 | [PHASE1_PI_READY.md](PHASE1_PI_READY.md) | Headless + throttle + systemd + log rotation. Rủi ro thấp, làm ngay | Trước khi có Pi |
| 3 | [PHASE2_ABSTRACTION_VALIDATION.md](PHASE2_ABSTRACTION_VALIDATION.md) | Abstraction layer + A/B validation harness. Gate quyết định có migrate không | Sau Phase 1 ổn định |
| 4 | [PHASE3_YUNET_SFACE_MIGRATION.md](PHASE3_YUNET_SFACE_MIGRATION.md) | Migration YuNet+SFace, rebuild encodings, bỏ dlib | Chỉ khi Phase 2 gate PASS |
| 5 | [PHASE4_BENCHMARK_TUNE.md](PHASE4_BENCHMARK_TUNE.md) | Benchmark Pi5 thật + tune cosine thresholds | Trên Pi5 thật |

---

## Quyết Định Đã Chốt

- **Hardware:** Raspberry Pi 5 (4GB), active cooling, NVMe/SSD
- **Stack target:** YuNet (detector) + SFace (embedder) — cả hai từ OpenCV Zoo, INT8 ONNX
- **Deps cuối:** chỉ `opencv-python-headless` + 2 file ONNX (không cần dlib build trên ARM)
- **KHÔNG đụng:** MotionGuard (probe burst), RecognitionStabilizer, NotificationWorker, services

---

## Quick Reference: Config Keys Quan Trọng

```bash
# Pi 5 production (.env)
HEADLESS=true
FRAME_WIDTH=640
FRAME_HEIGHT=480
ACTIVE_PROCESS_EVERY=2
FACE_BACKEND=sface
TOLERANCE_SFACE=0.60          # retune sau Phase 4
CONFUSION_MARGIN_SFACE=0.12   # retune sau Phase 4
MOTION_MAX_IDLE_SEC=60
IDLE_PROBE_BURST=5
```

---

## Nguyên Tắc An Toàn

1. **Phase 1 độc lập** — không đụng model, rollback = `git revert`
2. **Gate trước Phase 3** — chỉ migrate khi validation harness nói PASS
3. **Backup .pkl** trước khi rebuild encodings
4. **Accuracy > Speed** — không chấp nhận confusion người nhà dù có thêm bao nhiêu FPS
5. **Đo trên Pi thật** — không tin số benchmark từ NUC hay benchmark cộng đồng
