#!/usr/bin/env python3
"""
Export YOLO face detection model từ .pt sang .onnx để tăng tốc inference trên CPU.

ONNX Runtime nhanh hơn PyTorch ~1.5-2x trên CPU nhờ:
- Static graph (không có dynamic graph overhead của PyTorch)
- CPU-specific kernel optimizations (SIMD, threading pool)
- Không cần load toàn bộ PyTorch framework khi inference

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --model src/yolo/yolov11s-face.pt
    python src/export_onnx.py --model src/yolo/yolov11n-face.pt --imgsz 416
"""

import argparse
import os
import sys
import time

import config


def export_to_onnx(model_path: str, imgsz: int = 640, opset: int = 17) -> str:
    """
    Export YOLO .pt model sang .onnx.

    Args:
        model_path: Đường dẫn đến file .pt
        imgsz: Kích thước ảnh input (pixel), dùng cùng giá trị với YOLO_INPUT_WIDTH
        opset: ONNX opset version (17 tương thích rộng nhất)

    Returns:
        Đường dẫn đến file .onnx đã tạo
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics chưa được cài. Chạy: pip install ultralytics")
        sys.exit(1)

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("ERROR: onnxruntime chưa được cài. Chạy: pip install onnxruntime")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Không tìm thấy model: {model_path}")
        sys.exit(1)

    onnx_path = model_path.replace(".pt", ".onnx")

    if os.path.exists(onnx_path):
        print(f"File ONNX đã tồn tại: {onnx_path}")
        overwrite = input("Ghi đè? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("Bỏ qua export.")
            return onnx_path

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Exporting sang ONNX (imgsz={imgsz}, opset={opset})...")
    start = time.time()

    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,   # ONNX simplifier: loại bỏ node dư thừa
        dynamic=False,   # Fixed shape: tốt hơn cho inference tốc độ cao
        half=False,      # FP32 (CPU không hỗ trợ FP16 native)
    )

    elapsed = time.time() - start
    print(f"Export xong trong {elapsed:.1f}s")
    print(f"ONNX model: {exported}")

    # Benchmark nhanh: so sánh tốc độ .pt vs .onnx
    _run_benchmark(model_path, str(exported), imgsz)

    return str(exported)


def _run_benchmark(pt_path: str, onnx_path: str, imgsz: int, n_runs: int = 20):
    """So sánh tốc độ inference PyTorch vs ONNX Runtime."""
    import numpy as np
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    print(f"\nBenchmark ({n_runs} lần chạy, imgsz={imgsz})...")

    # Tạo ảnh test ngẫu nhiên
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # PyTorch
    pt_model = YOLO(pt_path)
    pt_model(dummy, verbose=False)  # warmup
    t0 = time.time()
    for _ in range(n_runs):
        pt_model(dummy, verbose=False)
    pt_ms = (time.time() - t0) / n_runs * 1000

    # ONNX Runtime
    onnx_model = YOLO(onnx_path)
    onnx_model(dummy, verbose=False)  # warmup
    t0 = time.time()
    for _ in range(n_runs):
        onnx_model(dummy, verbose=False)
    onnx_ms = (time.time() - t0) / n_runs * 1000

    speedup = pt_ms / onnx_ms if onnx_ms > 0 else 0
    print(f"\n  PyTorch:       {pt_ms:.1f} ms/frame")
    print(f"  ONNX Runtime:  {onnx_ms:.1f} ms/frame")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"\nGhi chú: Để dùng model ONNX, cập nhật config.py:")
    print(f'  MODEL_PATH = "{onnx_path}"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt → .onnx để tăng tốc trên CPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL_PATH,
        help="Đường dẫn đến file .pt"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.YOLO_INPUT_WIDTH,
        help="Kích thước ảnh input (nên dùng cùng giá trị với YOLO_INPUT_WIDTH trong config.py)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    onnx_path = export_to_onnx(args.model, args.imgsz, args.opset)
    print(f"\nDone: {onnx_path}")
