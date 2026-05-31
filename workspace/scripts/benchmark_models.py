#!/usr/bin/env python3
"""
YOLO Face Detection Benchmark
Đánh giá tất cả model .pt trong src/yolo/ trên các video trong data/

Usage:
    python src/benchmark_models.py
    python src/benchmark_models.py --frames 150 --sample 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_SRC_DIR   = Path(__file__).parent
_YOLO_DIR  = _SRC_DIR / "yolo"
_DATA_DIR  = _SRC_DIR.parent / "data"
_OUT_DIR   = _SRC_DIR.parent / "benchmark_results"

WARMUP_FRAMES     = 5     # bỏ qua N frame đầu (JIT warm-up)
CONF_THRESHOLD    = 0.15  # ngưỡng confidence để tính là "có mặt"
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv"}

# ANSI colors
_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_R  = "\033[91m"   # red
_B  = "\033[94m"   # blue
_W  = "\033[97m"   # white bold
_NC = "\033[0m"    # reset


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    detected:   bool
    conf:       float    # best confidence score (0 if no face)
    face_px:    int      # min(w,h) of best face bbox (0 if no face)
    n_faces:    int      # total faces detected
    ms:         float    # inference time (ms)


@dataclass
class ModelReport:
    model_name:    str
    model_size_mb: float
    status:        str        # "ok" | "error" | "incompatible"
    error_msg:     str = ""
    results:       List[FrameResult] = field(default_factory=list)

    # Derived stats (filled by compute_stats)
    fps:           float = 0.0
    avg_ms:        float = 0.0
    p95_ms:        float = 0.0
    detect_rate:   float = 0.0   # % frames with face
    avg_conf:      float = 0.0
    avg_face_px:   float = 0.0

    def compute_stats(self) -> None:
        if not self.results:
            return
        times = [r.ms for r in self.results]
        self.avg_ms      = float(np.mean(times))
        self.p95_ms      = float(np.percentile(times, 95))
        self.fps         = 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0
        detected         = [r for r in self.results if r.detected]
        self.detect_rate = len(detected) / len(self.results) * 100
        if detected:
            self.avg_conf    = float(np.mean([r.conf    for r in detected]))
            self.avg_face_px = float(np.mean([r.face_px for r in detected]))


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_paths: List[Path], max_frames: int, sample_every: int) -> List[np.ndarray]:
    """Trích xuất frames từ tất cả video, dùng chung cho mọi model (fair comparison)."""
    frames: List[np.ndarray] = []
    quota = max(1, max_frames // max(len(video_paths), 1))

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  {_R}[WARN] Không mở được: {vp.name}{_NC}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step  = max(sample_every, total // quota) if total > 0 else sample_every
        count = 0
        idx   = 0

        while count < quota:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
            idx   += step

        cap.release()
        print(f"  Video {vp.name:<35} → {count} frames lấy mẫu")

    return frames


# ---------------------------------------------------------------------------
# Single model benchmark
# ---------------------------------------------------------------------------

def run_model(model_path: Path, frames: List[np.ndarray]) -> ModelReport:
    """Chạy benchmark một model trên danh sách frames."""
    size_mb = model_path.stat().st_size / 1024 / 1024
    report  = ModelReport(model_name=model_path.name, model_size_mb=size_mb, status="error")

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        # Verify model loads with a dummy inference
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        model(dummy, verbose=False)
    except Exception as e:
        report.status    = "incompatible"
        report.error_msg = str(e)[:80]
        return report

    # Warm-up (JIT compilation, caching)
    warmup_frames = frames[:WARMUP_FRAMES] if len(frames) >= WARMUP_FRAMES else frames
    for wf in warmup_frames:
        try:
            model(wf, verbose=False)
        except Exception:
            pass

    # Actual benchmark
    report.status = "ok"
    for frame in frames:
        t0 = time.perf_counter()
        try:
            results = model(frame, verbose=False)
        except Exception:
            report.results.append(FrameResult(False, 0.0, 0, 0, 0.0))
            continue
        ms = (time.perf_counter() - t0) * 1000

        best_conf = 0.0
        best_px   = 0
        n_faces   = 0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                n_faces += 1
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    best_px = min(abs(x2 - x1), abs(y2 - y1))

        report.results.append(FrameResult(
            detected=n_faces > 0,
            conf=best_conf,
            face_px=best_px,
            n_faces=n_faces,
            ms=ms,
        ))

    report.compute_stats()
    return report


# ---------------------------------------------------------------------------
# Save sample detections
# ---------------------------------------------------------------------------

def save_samples(reports: List[ModelReport], frames: List[np.ndarray], out_dir: Path) -> None:
    """Lưu ảnh minh họa detection của mỗi model vào benchmark_results/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    # Pick a frame that likely has a face (middle of the list)
    sample_frame = frames[len(frames) // 2] if frames else None
    if sample_frame is None:
        return

    for report in reports:
        if report.status != "ok":
            continue
        try:
            model  = YOLO(str(_YOLO_DIR / report.model_name))
            canvas = sample_frame.copy()
            results = model(canvas, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(canvas, f"{conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

            label = report.model_name.replace(".pt", "")
            cv2.putText(canvas, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            out_path = out_dir / f"{label}.jpg"
            cv2.imwrite(str(out_path), canvas)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scoring & recommendation
# ---------------------------------------------------------------------------

def _score(r: ModelReport) -> float:
    """
    Score tổng hợp để xếp hạng model cho hệ thống này (CPU-only):
      - detect_rate   35%  (quan trọng nhất — phải detect được mặt)
      - fps           40%  (CPU-only nên tốc độ rất quan trọng)
      - avg_conf      15%
      - avg_face_px   10%
    """
    if r.status != "ok" or r.fps == 0:
        return 0.0
    fps_score      = min(r.fps / 30.0, 1.0)           # chuẩn hóa: 30fps = 1.0
    detect_score   = r.detect_rate / 100.0
    conf_score     = r.avg_conf
    face_px_score  = min(r.avg_face_px / 300.0, 1.0)  # chuẩn hóa: 300px = 1.0
    return (0.40 * fps_score + 0.35 * detect_score
            + 0.15 * conf_score + 0.10 * face_px_score)


def _fps_color(fps: float) -> str:
    if fps >= 20: return _G
    if fps >= 10: return _Y
    return _R


def _rate_color(rate: float) -> str:
    if rate >= 75: return _G
    if rate >= 50: return _Y
    return _R


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(reports: List[ModelReport], n_frames: int) -> None:
    ok = [r for r in reports if r.status == "ok"]
    if ok:
        ok.sort(key=_score, reverse=True)
        best = ok[0]
    else:
        best = None

    print()
    print(f"{_W}{'═'*88}{_NC}")
    print(f"{_W}{'  YOLO FACE DETECTION BENCHMARK':^88}{_NC}")
    print(f"{_W}{'═'*88}{_NC}")
    print(f"  Frames kiểm tra : {n_frames}")
    print(f"  Confidence ngưỡng: {CONF_THRESHOLD}")
    print(f"  Hardware         : Intel i7-8650U  |  CPU-only (no GPU)")
    print(f"{_W}{'─'*88}{_NC}")

    # Header
    print(f"  {'Model':<30} {'Size':>6}  {'FPS':>6}  {'ms/frame':>8}  "
          f"{'p95ms':>6}  {'Detect%':>7}  {'Conf':>5}  {'FacePx':>6}  {'Score':>5}")
    print(f"  {'─'*30} {'─'*6}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*5}")

    for r in reports:
        if r.status == "ok":
            fc   = _fps_color(r.fps)
            rc   = _rate_color(r.detect_rate)
            star = f" {_G}★ BEST{_NC}" if best and r.model_name == best.model_name else ""
            print(
                f"  {r.model_name:<30} {r.model_size_mb:>5.1f}M  "
                f"{fc}{r.fps:>6.1f}{_NC}  {r.avg_ms:>7.1f}ms  {r.p95_ms:>5.1f}ms  "
                f"{rc}{r.detect_rate:>6.1f}%{_NC}  {r.avg_conf:>5.2f}  "
                f"{r.avg_face_px:>5.0f}px  {_score(r):>5.2f}"
                f"{star}"
            )
        else:
            tag = "INCOMPATIBLE" if r.status == "incompatible" else "ERROR"
            print(f"  {r.model_name:<30} {r.model_size_mb:>5.1f}M  "
                  f"{_R}[{tag}]{_NC}  {r.error_msg[:40]}")

    print(f"{_W}{'─'*88}{_NC}")

    # Legend
    print(f"\n  Màu FPS  : {_G}xanh ≥ 20fps{_NC}  {_Y}vàng ≥ 10fps{_NC}  {_R}đỏ < 10fps{_NC}")
    print(f"  Màu Detect%: {_G}xanh ≥ 75%{_NC}  {_Y}vàng ≥ 50%{_NC}  {_R}đỏ < 50%{_NC}")
    print(f"  Score: tổng hợp (FPS 40% + Detect 35% + Conf 15% + FaceSize 10%)")

    # Recommendation
    print(f"\n{_W}{'─'*88}{_NC}")
    print(f"{_W}  KHUYẾN NGHỊ{_NC}")
    print(f"{'─'*88}")

    if not ok:
        print(f"  {_R}Không có model nào chạy được.{_NC}")
        return

    # Best overall
    print(f"  {_G}★ Tổng thể tốt nhất : {best.model_name}{_NC}")
    print(f"    → {best.fps:.1f} FPS  |  detect {best.detect_rate:.0f}%  |  score {_score(best):.2f}")

    # Best for real-time (≥20fps)
    realtime = [r for r in ok if r.fps >= 20]
    if realtime:
        rt_best = max(realtime, key=lambda r: r.detect_rate)
        if rt_best.model_name != best.model_name:
            print(f"\n  {_Y}★ Real-time (≥20fps): {rt_best.model_name}{_NC}")
            print(f"    → {rt_best.fps:.1f} FPS  |  detect {rt_best.detect_rate:.0f}%")

    # Best accuracy (highest detect rate)
    acc_best = max(ok, key=lambda r: (r.detect_rate, r.avg_conf))
    if acc_best.model_name != best.model_name:
        print(f"\n  {_B}★ Detect mặt nhiều nhất: {acc_best.model_name}{_NC}")
        print(f"    → detect {acc_best.detect_rate:.0f}%  |  {acc_best.fps:.1f} FPS")

    # Training vs Recognition recommendation
    print(f"\n  Gợi ý theo use-case:")
    training_pick = max(ok, key=lambda r: r.detect_rate)
    recog_pick    = max(realtime, key=lambda r: r.detect_rate) if realtime else best
    print(f"    Thu thập training data  → {_G}{training_pick.model_name}{_NC}"
          f"  (detect nhiều nhất, tốc độ không quan trọng)")
    print(f"    Nhận diện real-time     → {_G}{recog_pick.model_name}{_NC}"
          f"  (cân bằng tốc độ + độ chính xác)")

    print(f"{'─'*88}")
    print(f"  {_B}Ảnh minh họa đã lưu tại: {_OUT_DIR}/{_NC}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark tất cả YOLO model trong src/yolo/ trên video trong data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames",  type=int, default=100,
                   help="Tổng số frames lấy mẫu (chia đều cho các video)")
    p.add_argument("--sample",  type=int, default=10,
                   help="Lấy 1 frame mỗi N frames khi scan video")
    p.add_argument("--yolo-dir", type=str, default=str(_YOLO_DIR),
                   help="Thư mục chứa các file .pt")
    p.add_argument("--data-dir", type=str, default=str(_DATA_DIR),
                   help="Thư mục chứa video")
    p.add_argument("--no-save", action="store_true",
                   help="Không lưu ảnh minh họa")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yolo_dir = Path(args.yolo_dir)
    data_dir = Path(args.data_dir)

    # Discover models
    model_paths = sorted(yolo_dir.glob("*.pt"))
    if not model_paths:
        print(f"{_R}Không tìm thấy file .pt trong {yolo_dir}{_NC}")
        sys.exit(1)

    # Discover videos
    video_paths = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not video_paths:
        print(f"{_R}Không tìm thấy video trong {data_dir}{_NC}")
        sys.exit(1)

    print(f"\n{_W}  YOLO Face Detection Benchmark{_NC}")
    print(f"  Models  : {len(model_paths)} file .pt trong {yolo_dir}")
    print(f"  Videos  : {len(video_paths)} file trong {data_dir}")
    print(f"  Frames  : tối đa {args.frames} frames (sample mỗi {args.sample} frame)\n")

    # ── Step 1: Extract frames ───────────────────────────────────────────────
    print(f"{_W}[1/3] Trích xuất frames mẫu...{_NC}")
    frames = sample_frames(video_paths, args.frames, args.sample)
    if not frames:
        print(f"{_R}Không trích xuất được frame nào.{_NC}")
        sys.exit(1)
    print(f"  → Tổng: {len(frames)} frames\n")

    # ── Step 2: Benchmark each model ────────────────────────────────────────
    print(f"{_W}[2/3] Chạy benchmark...{_NC}")
    reports: List[ModelReport] = []

    for mp in model_paths:
        print(f"  {mp.name:<35}", end="", flush=True)
        report = run_model(mp, frames)
        reports.append(report)

        if report.status == "ok":
            print(f"{_G}OK{_NC}  "
                  f"{report.fps:5.1f} FPS  "
                  f"detect {report.detect_rate:5.1f}%  "
                  f"score {_score(report):.2f}")
        elif report.status == "incompatible":
            print(f"{_Y}INCOMPATIBLE{_NC}  {report.error_msg[:50]}")
        else:
            print(f"{_R}ERROR{_NC}  {report.error_msg[:50]}")

    # ── Step 3: Save samples ────────────────────────────────────────────────
    if not args.no_save:
        print(f"\n{_W}[3/3] Lưu ảnh minh họa...{_NC}")
        save_samples(reports, frames, _OUT_DIR)
        print(f"  → {_OUT_DIR}")

    # ── Print report ────────────────────────────────────────────────────────
    print_report(reports, len(frames))


if __name__ == "__main__":
    main()
