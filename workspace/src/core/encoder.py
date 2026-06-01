#!/usr/bin/env python3
"""
Encoding utility — rebuild face encodings from a dataset folder.

Supports two backends (controlled by config.FACE_BACKEND or explicit parameter):
  "dlib"  — YOLO + face_recognition (dlib ResNet 128-d, Euclidean distance)
  "sface" — YuNet + SFace (OpenCV-native, cosine distance, INT8 ONNX)

Used by training/data_collection.py and face_manager.py.
"""

import logging
import os
import pickle
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from core.detector import FaceDetection, FaceDetector
from core.embedder import FaceEmbedder


def _build_backend(backend: str) -> Tuple[FaceDetector, FaceEmbedder]:
    """Create (detector, embedder) pair for encoding."""
    if backend == "sface":
        from core.detector import YuNetDetector
        from core.embedder import SFaceEmbedder
        detector = YuNetDetector(model_path=config.YUNET_MODEL_PATH)
        embedder = SFaceEmbedder(model_path=config.SFACE_MODEL_PATH)
    else:
        from core.detector import YoloDetector
        from core.embedder import DlibEmbedder
        detector = YoloDetector(
            model_path=config.MODEL_PATH,
            detection_confidence=config.ENCODING_CONFIDENCE,
            input_width=config.YOLO_INPUT_WIDTH,
        )
        embedder = DlibEmbedder()
    return detector, embedder


def _extract_face_encoding(
    image: np.ndarray,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    logger: logging.Logger,
    encoding_padding: int = config.ENCODING_PADDING,
) -> Optional[np.ndarray]:
    """Extract a face embedding using the given detector + embedder.

    Falls back to HOG full-image scan (dlib only) when detector misses.
    """
    detections = detector.detect(image)

    if detections:
        best = max(detections, key=lambda d: d.confidence)
        enc = embedder.encode(image, best, padding=encoding_padding)
        if enc is not None:
            return enc

    # HOG fallback (dlib only — SFace has no meaningful fallback)
    try:
        import face_recognition as _fr
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encs = _fr.face_encodings(rgb)
        if encs:
            logger.debug("HOG fallback used for encoding (detector missed)")
            return encs[0]
    except ImportError:
        pass
    except Exception:
        pass

    return None


def _cluster_to_max(
    encodings: List[np.ndarray],
    max_count: int,
) -> List[np.ndarray]:
    """Greedy diversity-based downsampling to at most max_count encodings.

    Keeps each encoding only if its minimum distance to the already-selected
    set exceeds CLUSTERING_THRESHOLD. Falls back to evenly-spaced fill.
    """
    if len(encodings) <= max_count:
        return list(encodings)

    encs = np.array(encodings)   # (N, D)
    selected: List[int] = [0]

    for i in range(1, len(encs)):
        if len(selected) >= max_count:
            break
        dists = np.linalg.norm(encs[selected] - encs[i], axis=1)
        if float(dists.min()) > config.CLUSTERING_THRESHOLD:
            selected.append(i)

    if len(selected) < max_count:
        present   = set(selected)
        remaining = [i for i in range(len(encs)) if i not in present]
        needed    = max_count - len(selected)
        step      = max(1, len(remaining) // needed)
        selected.extend(remaining[::step][:needed])

    return [encodings[i] for i in selected[:max_count]]


def update_person_encodings(
    person_name: str,
    new_image_paths: List[str],
    encodings_path: str,
    model_path: str,
    backend: str = config.FACE_BACKEND,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """Incrementally update encodings for ONE person without touching others.

    Encodes only new_image_paths, merges with existing encodings for this
    person, clusters down to MAX_ENCODINGS_PER_PERSON, then writes back.

    Returns:
        (new_encodings_added, total_encodings_for_person)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not new_image_paths:
        logger.info("No new images for %s — nothing to update", person_name)
        return 0, 0

    logger.info(
        "Incremental update '%s': encoding %d new images (backend=%s)…",
        person_name, len(new_image_paths), backend,
    )

    detector, embedder = _build_backend(backend)
    new_encs: List[np.ndarray] = []

    for path in new_image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        enc = _extract_face_encoding(img, detector, embedder, logger)
        if enc is not None:
            new_encs.append(enc)

    logger.info(
        "%s: %d / %d images → valid encodings",
        person_name, len(new_encs), len(new_image_paths),
    )
    if not new_encs:
        logger.warning("No valid encodings extracted for %s", person_name)
        return 0, 0

    # Load existing pkl — separate this person from everyone else
    other_encs: List[np.ndarray] = []
    other_names: List[str] = []
    existing_person_encs: List[np.ndarray] = []

    if os.path.exists(encodings_path):
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        for enc, name in zip(data["encodings"], data["names"]):
            if name == person_name:
                existing_person_encs.append(enc)
            else:
                other_encs.append(enc)
                other_names.append(name)

    combined  = existing_person_encs + new_encs
    clustered = _cluster_to_max(combined, config.MAX_ENCODINGS_PER_PERSON)

    # Backup then save
    if os.path.exists(encodings_path):
        backup = encodings_path.replace(".pkl", "_backup.pkl")
        try:
            os.replace(encodings_path, backup)
            logger.info("Backed up → %s", backup)
        except OSError as e:
            logger.warning("Backup failed: %s", e)

    os.makedirs(os.path.dirname(encodings_path) or ".", exist_ok=True)
    with open(encodings_path, "wb") as f:
        pickle.dump(
            {
                "encodings": other_encs + clustered,
                "names":     other_names + [person_name] * len(clustered),
                "backend":   backend,
            },
            f,
        )

    logger.info(
        "%s: %d existing + %d new → %d after clustering | %d other persons",
        person_name,
        len(existing_person_encs),
        len(new_encs),
        len(clustered),
        len(set(other_names)),
    )
    return len(new_encs), len(clustered)


def rebuild_encodings(
    dataset_path: str,
    model_path: str,
    encodings_path: str,
    backend: str = config.FACE_BACKEND,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, int]:
    """Rebuild face encodings from scratch for every person in dataset_path.

    Reads all images under ``dataset_path/<person>/``, extracts embeddings,
    clusters per-person to MAX_ENCODINGS_PER_PERSON, and writes the .pkl.

    Args:
        dataset_path:   Root folder — one sub-folder per person.
        model_path:     YOLO model path (used only when backend="dlib").
        encodings_path: Destination .pkl file.
        backend:        "dlib" or "sface".
        logger:         Optional; creates default if omitted.

    Returns:
        (total_encodings, total_persons)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(dataset_path):
        logger.error("Dataset path not found: %s", dataset_path)
        return 0, 0

    logger.info("Rebuilding encodings (backend=%s)…", backend)
    detector, embedder = _build_backend(backend)

    all_encodings: List[np.ndarray] = []
    all_names:     List[str]         = []
    persons_processed = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        image_files = [
            f for f in os.listdir(person_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not image_files:
            continue

        person_encs: List[np.ndarray] = []
        for fname in image_files:
            img = cv2.imread(os.path.join(person_folder, fname))
            if img is None:
                continue
            enc = _extract_face_encoding(img, detector, embedder, logger)
            if enc is not None:
                person_encs.append(enc)

        clustered = _cluster_to_max(person_encs, config.MAX_ENCODINGS_PER_PERSON)
        all_encodings.extend(clustered)
        all_names.extend([person_name] * len(clustered))

        logger.info(
            "  %s: %d / %d images → %d encodings",
            person_name, len(person_encs), len(image_files), len(clustered),
        )
        persons_processed += 1

    # Backup existing .pkl before overwriting
    if os.path.exists(encodings_path):
        backup = encodings_path.replace(".pkl", "_backup.pkl")
        try:
            os.replace(encodings_path, backup)
            logger.info("Previous encodings backed up → %s", backup)
        except OSError as e:
            logger.warning("Backup failed: %s", e)

    os.makedirs(os.path.dirname(encodings_path) or ".", exist_ok=True)
    with open(encodings_path, "wb") as f:
        pickle.dump(
            {
                "encodings": all_encodings,
                "names":     all_names,
                "backend":   backend,
            },
            f,
        )

    logger.info(
        "Saved: %d encodings for %d persons (backend=%s) → %s",
        len(all_encodings), persons_processed, backend, encodings_path,
    )
    return len(all_encodings), persons_processed
