#!/usr/bin/env python3
"""
Centralized configuration for the face recognition system.
All paths are resolved relative to this file's location — no hardcoded absolute paths.
"""

import os

# --- Directory layout ---
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SRC_DIR)

# Data paths
DATASET_PATH = os.path.join(_WORKSPACE_DIR, "family_images")
MODEL_PATH = os.path.join(_SRC_DIR, "yolo", "yolov11n-face.pt")
ENCODINGS_DIR = os.path.join(_WORKSPACE_DIR, "model")

# --- Recognition parameters ---
TOLERANCE = 0.6              # Face matching tolerance (lower = stricter)
DETECTION_CONFIDENCE = 0.3   # Min YOLO confidence for live detection (low = catches distant faces)
ENCODING_CONFIDENCE = 0.5    # Min YOLO confidence when building encodings from dataset
RECOGNITION_PADDING = 15     # Pixel padding around detected face for recognition
ENCODING_PADDING = 20        # Pixel padding around detected face when encoding training images
MAX_FPS_SAMPLES = 30         # Rolling window size for FPS calculation

# --- Learning / training parameters ---
# Detection threshold: how confident YOLO must be before a face region is processed.
# Lower → catches faces in poor lighting or non-frontal angles.
LEARNING_DETECTION_THRESHOLD = 0.4   # was 0.7 — relaxed to support varied conditions
# Capture threshold: minimum *overall* quality score (0-1) required to save a sample.
# Lower → accepts darker / more varied images; higher → stricter quality gate.
LEARNING_CAPTURE_THRESHOLD = 0.25    # was hardcoded 0.5 in run_learning_mode
MIN_FACE_SIZE = 100                  # Min face dimension (px) to be considered usable
MAX_SAMPLES_PER_PERSON = 50         # Max images collected per person slot
CAPTURE_INTERVAL = 0.5              # Seconds between auto-captures to avoid duplicates
LEARNING_PADDING = 20               # Pixel padding during face extraction in learning mode

# --- Optimization parameters ---
# YOLO confidence gate for extracting encodings.
# Lowered from 0.7 → 0.35 so dark/angled/blurry images are NOT silently rejected.
# A face_recognition fallback is used when YOLO confidence is below this value.
HIGH_CONFIDENCE_THRESHOLD = 0.35
# Minimum quality score to keep an encoding.
# Lowered from 0.6 → 0.15 so adverse-condition images (dark, blurry) pass through.
OPTIMIZE_QUALITY_THRESHOLD = 0.15
MAX_ENCODINGS_PER_PERSON = 30      # Max encodings kept per person after clustering (was 20)
CLUSTERING_THRESHOLD = 0.15        # Face distance below which two encodings are near-duplicates (was 0.4 → 0.5)
                                   # Typical same-person distances: 0.2-0.5, so 0.15 only deduplicates
                                   # near-identical shots without collapsing different-condition images.
FACE_AREA_NORMALIZATION = 5000     # Divisor when converting face area to a 0-1 quality score (was 10000)
OPTIMIZE_PADDING = 20              # Pixel padding during face extraction in optimizer

# --- Frame extraction parameters (training data collection from video) ---
# mp4v codec compresses Laplacian variance from ~200+ (raw) down to 6-43.
# These thresholds are calibrated for compressed video, not live camera frames.
FRAME_MIN_LAPLACIAN = 15       # Min Laplacian variance; keeps ~top-70% of compressed frames
FRAME_MIN_FACE_PX = 70         # Min face dimension in px; lower than MIN_FACE_SIZE to allow angled faces
FRAME_DETECT_CONF = 0.15       # Min YOLO confidence; lower catches non-frontal / side-profile faces

# --- Camera / display ---
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
