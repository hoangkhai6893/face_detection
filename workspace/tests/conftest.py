"""
Shared pytest fixtures for the training manager test suite.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """200×200 BGR image with moderate brightness and sharpness."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Gradient background (gives Laplacian variance)
    for i in range(200):
        img[i, :] = [int(i * 0.8), int(i * 0.5), int(i * 0.3)]
    # High-contrast circle to boost Laplacian variance
    cv2.circle(img, (100, 100), 60, (200, 180, 160), -1)
    cv2.circle(img, (100, 100), 60, (20, 20, 20), 2)
    cv2.circle(img, (80, 90), 10, (30, 30, 30), -1)   # "eye"
    cv2.circle(img, (120, 90), 10, (30, 30, 30), -1)  # "eye"
    return img


@pytest.fixture
def blurry_image(synthetic_face_image) -> np.ndarray:
    """Heavily blurred version — Laplacian variance << 100."""
    return cv2.GaussianBlur(synthetic_face_image, (31, 31), 0)


@pytest.fixture
def dark_image(synthetic_face_image) -> np.ndarray:
    """Very dark version — brightness < 0.10."""
    return np.clip(synthetic_face_image.astype(np.float32) * 0.05, 0, 255).astype(np.uint8)


@pytest.fixture
def overexposed_image(synthetic_face_image) -> np.ndarray:
    """Overexposed version — brightness > 0.92."""
    return np.clip(synthetic_face_image.astype(np.float32) * 5.0, 0, 255).astype(np.uint8)


@pytest.fixture
def tiny_face_image() -> np.ndarray:
    """50×50 image — below MIN_FACE_SIZE (100)."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    cv2.circle(img, (25, 25), 18, (200, 180, 160), -1)
    return img


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dataset(tmp_path, synthetic_face_image) -> Path:
    """Temporary dataset directory with two persons and 3 images each."""
    for person in ("Alice", "Bob"):
        person_dir = tmp_path / person
        person_dir.mkdir()
        for i in range(3):
            img_path = person_dir / f"{person}_{i}.jpg"
            cv2.imwrite(str(img_path), synthetic_face_image)
    return tmp_path


# ---------------------------------------------------------------------------
# Video fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_video(tmp_path, synthetic_face_image) -> Path:
    """Short synthetic MP4 (30 frames, 30 fps) containing the face image."""
    out_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (200, 200))
    for _ in range(30):
        writer.write(synthetic_face_image)
    writer.release()
    return out_path


@pytest.fixture
def blurry_video(tmp_path, blurry_image) -> Path:
    """All-blurry 30-frame video."""
    out_path = tmp_path / "blurry_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (200, 200))
    for _ in range(30):
        writer.write(blurry_image)
    writer.release()
    return out_path
