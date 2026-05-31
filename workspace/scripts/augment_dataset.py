#!/usr/bin/env python3
"""
Dataset Augmentation Tool — tự sinh dữ liệu huấn luyện đa dạng.

Đọc ảnh gốc từ family_images/<person>/, áp dụng 8 loại biến đổi bất lợi
(blur, low-light, noise, compression artifacts, …) và lưu từng biến thể
cạnh ảnh gốc.  Sau đó tự động rebuild encodings để hệ thống nhận diện
tốt hơn trong các điều kiện khó.

Cách hoạt động (không fine-tune weights):
  - face_recognition dùng model dlib pre-trained, không thể train lại.
  - "Self-training" ở đây là: sinh ảnh bất lợi → add vào dataset → re-encode.
  - Khi recognition gặp frame tối/nhòe, encoding gần với training (cũng tối/nhòe)
    → face distance thấp hơn → nhận diện chính xác hơn.

Usage:
    python augment_dataset.py                     # toàn bộ dataset
    python augment_dataset.py --person Khai       # 1 người
    python augment_dataset.py --no-preview        # bỏ qua preview
    python augment_dataset.py --no-encode         # không rebuild sau
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from core.encoder import rebuild_encodings


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def _aug_blur_mild(img: np.ndarray) -> np.ndarray:
    """Gaussian blur nhẹ — hơi mất nét."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def _aug_blur_heavy(img: np.ndarray) -> np.ndarray:
    """Gaussian blur mạnh — nhòe nặng."""
    return cv2.GaussianBlur(img, (21, 21), 0)


def _aug_motion_blur(img: np.ndarray) -> np.ndarray:
    """Motion blur ngang — giả lập camera rung."""
    size = 15
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)


def _aug_low_light_mild(img: np.ndarray) -> np.ndarray:
    """Giảm độ sáng xuống 50% — thiếu sáng nhẹ."""
    return np.clip(img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)


def _aug_low_light_heavy(img: np.ndarray) -> np.ndarray:
    """Giảm độ sáng xuống 25% — thiếu sáng nặng."""
    return np.clip(img.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)


def _aug_noise(img: np.ndarray) -> np.ndarray:
    """Thêm Gaussian noise σ=30 — ảnh nhiễu/grainy."""
    noise = np.random.normal(0, 30, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _aug_low_quality(img: np.ndarray) -> np.ndarray:
    """Re-encode JPEG quality=15 — giả lập camera chất lượng thấp."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _aug_combined(img: np.ndarray) -> np.ndarray:
    """Kết hợp blur nhẹ + low-light nặng — điều kiện tổng hợp khó nhất."""
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    return np.clip(blurred.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


# Registry: key = suffix in filename, value = function
AUGMENTATIONS: Dict[str, callable] = {
    "blur_mild":       _aug_blur_mild,
    "blur_heavy":      _aug_blur_heavy,
    "motion_blur":     _aug_motion_blur,
    "low_light_mild":  _aug_low_light_mild,
    "low_light_heavy": _aug_low_light_heavy,
    "noise":           _aug_noise,
    "low_quality":     _aug_low_quality,
    "combined":        _aug_combined,
}


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

_CELL_SIZE = (160, 140)   # (width, height) of each cell in the preview grid


def _make_preview_collage(original: np.ndarray) -> np.ndarray:
    """
    Build a preview image showing the original + all augmented variants.
    Layout: 3 columns × 3 rows (9 cells: 1 original + 8 augmentations).
    """
    cols = 3
    rows = 3
    cell_w, cell_h = _CELL_SIZE

    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    cells = [("original", original)] + [
        (name, fn(original)) for name, fn in AUGMENTATIONS.items()
    ]

    for idx, (label, img) in enumerate(cells):
        row = idx // cols
        col = idx % cols
        y0, y1 = row * cell_h, (row + 1) * cell_h
        x0, x1 = col * cell_w, (col + 1) * cell_w

        # Fit image into cell preserving aspect ratio
        h, w = img.shape[:2]
        scale = min((cell_w - 4) / w, (cell_h - 20) / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))

        # Centre in cell
        py = (cell_h - 20 - nh) // 2
        px = (cell_w - nw) // 2
        canvas[y0 + py: y0 + py + nh, x0 + px: x0 + px + nw] = resized

        # Label at bottom of cell
        cv2.putText(
            canvas, label,
            (x0 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 200), 1,
        )

    return canvas


def preview_augmentations(sample_image: np.ndarray, person_name: str) -> bool:
    """
    Hiển thị collage preview tất cả augmentations trên 1 ảnh mẫu.

    Returns:
        True nếu user nhấn ENTER (xác nhận), False nếu nhấn ESC (hủy).
    """
    collage = _make_preview_collage(sample_image)

    header_h = 40
    header = np.zeros((header_h, collage.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"PREVIEW — {person_name}  |  ENTER = augment all   ESC = cancel",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1,
    )
    full = np.vstack([header, collage])

    win = "Augmentation Preview"
    cv2.imshow(win, full)
    cv2.moveWindow(win, 50, 50)

    confirmed = False
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, ord('y')):   # ENTER or y
            confirmed = True
            break
        elif key in (27, ord('n')): # ESC or n
            break

    cv2.destroyAllWindows()
    return confirmed


# ---------------------------------------------------------------------------
# Augmentation pipeline per person
# ---------------------------------------------------------------------------

def augment_person(
    person_name: str,
    dataset_path: str,
    show_preview: bool,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """
    Augment semua gambar untuk satu orang.

    Idempotent: skip augmented file yang sudah ada.

    Returns:
        (total_generated, total_skipped_existing)
    """
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        logger.warning("Folder not found: %s", person_folder)
        return 0, 0

    # Collect original images only (exclude already-augmented files)
    originals = sorted([
        f for f in os.listdir(person_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        and "_aug_" not in f
    ])

    if not originals:
        logger.warning("%s: no original images found, skipping.", person_name)
        return 0, 0

    logger.info("%s: %d original image(s) found.", person_name, len(originals))

    # Preview on first original image
    if show_preview:
        sample_path = os.path.join(person_folder, originals[0])
        sample_img = cv2.imread(sample_path)
        if sample_img is not None:
            confirmed = preview_augmentations(sample_img, person_name)
            if not confirmed:
                logger.info("%s: augmentation cancelled by user.", person_name)
                return 0, 0
        else:
            logger.warning("Could not read sample image for preview: %s", sample_path)

    generated = 0
    skipped = 0

    for fname in originals:
        stem = os.path.splitext(fname)[0]
        src_path = os.path.join(person_folder, fname)

        img = cv2.imread(src_path)
        if img is None:
            logger.warning("Cannot read: %s — skipping", fname)
            continue

        for aug_name, aug_fn in AUGMENTATIONS.items():
            dest_fname = f"{stem}_aug_{aug_name}.jpg"
            dest_path = os.path.join(person_folder, dest_fname)

            if os.path.exists(dest_path):
                skipped += 1
                continue

            try:
                augmented = aug_fn(img)
                if cv2.imwrite(dest_path, augmented):
                    generated += 1
                else:
                    logger.error("Failed to write: %s", dest_path)
            except Exception as e:
                logger.error("Augmentation '%s' failed on %s: %s", aug_name, fname, e)

    logger.info(
        "%s: generated %d | skipped (already exist) %d",
        person_name, generated, skipped,
    )
    return generated, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment face training dataset to improve recognition robustness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--person", "-p",
        default=None,
        metavar="NAME",
        help="Augment only this person. Omit to augment entire dataset.",
    )
    parser.add_argument(
        "--dataset",
        default=config.DATASET_PATH,
        help="Root dataset folder.",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_PATH,
        help="YOLO model path (used during re-encoding).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the visual preview step and augment immediately.",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Do not rebuild face encodings after augmentation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    args = parse_args()

    if not os.path.exists(args.dataset):
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    # Determine which persons to process
    if args.person:
        persons = [args.person]
    else:
        persons = sorted([
            d for d in os.listdir(args.dataset)
            if os.path.isdir(os.path.join(args.dataset, d))
        ])

    if not persons:
        logger.error("No person folders found in: %s", args.dataset)
        sys.exit(1)

    logger.info(
        "Starting augmentation for %d person(s): %s",
        len(persons), ", ".join(persons),
    )
    logger.info("Augmentation types: %s", ", ".join(AUGMENTATIONS.keys()))

    total_generated = 0
    total_skipped = 0

    for person in persons:
        gen, skip = augment_person(
            person_name=person,
            dataset_path=args.dataset,
            show_preview=not args.no_preview,
            logger=logger,
        )
        total_generated += gen
        total_skipped += skip

    logger.info(
        "Augmentation complete — total generated: %d | already existed (skipped): %d",
        total_generated, total_skipped,
    )

    if total_generated == 0 and total_skipped > 0:
        logger.info("All augmented files already exist. Dataset is up to date.")

    if args.no_encode:
        logger.info("Skipping encoding step (--no-encode).")
        return

    if total_generated == 0:
        logger.info("No new images generated — skipping encoding step.")
        return

    encodings_path = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
    logger.info("Rebuilding encodings: %s", encodings_path)
    total_enc, total_persons = rebuild_encodings(
        dataset_path=args.dataset,
        model_path=args.model,
        encodings_path=encodings_path,
        logger=logger,
    )
    logger.info(
        "Done — %d encodings for %d persons. Recognition system is up to date.",
        total_enc, total_persons,
    )


if __name__ == "__main__":
    main()
