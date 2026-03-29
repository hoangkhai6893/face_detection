#!/usr/bin/env python3
"""
Shared encoding utility — rebuild face encodings from a dataset folder.

Used by both face_learning_mode.py (after webcam session) and
add_training_data.py (after file import) so encoding logic lives in one place.
"""

import logging
import os
import pickle
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

import config
from core.face_utils import extract_face_region


def _extract_face_encoding(
    image: np.ndarray,
    yolo_model: YOLO,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Extract a 128-d face encoding from *image* using YOLO + face_recognition.
    Falls back to pure face_recognition if YOLO finds nothing confident enough.
    """
    results = yolo_model(image, verbose=False)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        confidences = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        best_box = result.boxes[best_idx]

        if float(best_box.conf) < config.ENCODING_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
        face_image = extract_face_region(image, x1, y1, x2, y2, config.ENCODING_PADDING)
        if face_image is None:
            continue

        try:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model="hog")
            if locations:
                encodings = face_recognition.face_encodings(rgb, locations)
                if encodings:
                    return encodings[0]
        except Exception:
            pass

    # Fallback: run face_recognition on the full image
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            return encodings[0]
    except Exception:
        pass

    return None


def rebuild_encodings(
    dataset_path: str,
    model_path: str,
    encodings_path: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """
    Rebuild face encodings from scratch for every person in *dataset_path*.

    Reads all images under ``dataset_path/<person>/``, extracts a 128-d
    encoding per image (YOLO-assisted), and writes the result to
    *encodings_path* as a pickle file.

    Args:
        dataset_path:   Root folder with one sub-folder per person.
        model_path:     Path to the YOLO .pt model file.
        encodings_path: Destination .pkl file for the encodings.
        logger:         Optional logger; a default one is created if omitted.

    Returns:
        (total_encodings, total_persons) — counts of what was written.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(dataset_path):
        logger.error("Dataset path not found: %s", dataset_path)
        return 0, 0

    logger.info("Loading YOLO model for encoding: %s", model_path)
    yolo_model = YOLO(model_path)

    all_encodings: List[np.ndarray] = []
    all_names: List[str] = []
    persons_processed = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        image_files = [
            f for f in os.listdir(person_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        if not image_files:
            continue

        ok = 0
        for fname in image_files:
            img = cv2.imread(os.path.join(person_folder, fname))
            if img is None:
                continue
            enc = _extract_face_encoding(img, yolo_model, logger)
            if enc is not None:
                all_encodings.append(enc)
                all_names.append(person_name)
                ok += 1

        logger.info("  %s: %d / %d images encoded", person_name, ok, len(image_files))
        persons_processed += 1

    # Backup existing encodings before overwriting
    if os.path.exists(encodings_path):
        backup = encodings_path.replace(".pkl", "_backup.pkl")
        try:
            os.replace(encodings_path, backup)
            logger.info("Previous encodings backed up to: %s", backup)
        except OSError as e:
            logger.warning("Could not backup encodings: %s", e)

    # Write new encodings
    os.makedirs(os.path.dirname(encodings_path) or ".", exist_ok=True)
    with open(encodings_path, "wb") as f:
        pickle.dump({"encodings": all_encodings, "names": all_names}, f)

    logger.info(
        "Encodings saved: %d encodings for %d persons → %s",
        len(all_encodings), persons_processed, encodings_path,
    )
    return len(all_encodings), persons_processed
