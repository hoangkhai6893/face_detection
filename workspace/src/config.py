#!/usr/bin/env python3
"""
Centralized configuration for the face recognition system.
All paths are resolved relative to this file's location — no hardcoded absolute paths.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass  # python-dotenv chưa cài — dùng env vars từ shell

# --- Directory layout ---
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SRC_DIR)

# Data paths
DATASET_PATH = os.path.join(_WORKSPACE_DIR, "family_images")
# Ưu tiên dùng ONNX Runtime (nhanh hơn PyTorch ~1.5-2x trên CPU).
# Chạy `python src/export_onnx.py` để tạo file .onnx lần đầu.
# Nếu chưa có .onnx, tự động fallback về .pt.
_ONNX_MODEL = os.path.join(_SRC_DIR, "yolo", "yolov11n-face.onnx")
_PT_MODEL   = os.path.join(_SRC_DIR, "yolo", "yolov11n-face.pt")
MODEL_PATH  = _ONNX_MODEL if os.path.exists(_ONNX_MODEL) else _PT_MODEL
ENCODINGS_DIR = os.path.join(_WORKSPACE_DIR, "model")

# --- Recognition parameters ---
TOLERANCE = 0.5              # Face matching tolerance (lower = stricter)
DETECTION_CONFIDENCE = 0.3   # Min YOLO confidence for live detection (low = catches distant faces)
ENCODING_CONFIDENCE = 0.5    # Min YOLO confidence when building encodings from dataset
RECOGNITION_PADDING = 15     # Pixel padding around detected face for recognition
ENCODING_PADDING = 20        # Pixel padding around detected face when encoding training images
MAX_FPS_SAMPLES = 30         # Rolling window size for FPS calculation
# Top-K voting: lấy K encoding gần nhất rồi vote theo tên người.
# Tránh trường hợp 1 encoding xấu trong DB kéo kết quả sai.
# K=1 = hành vi cũ (argmin thuần). K=5 khuyến nghị sau khi optimize (30 enc/người).
RECOGNITION_TOP_K = 5
# Nếu runner-up (người thứ 2) cách winner < margin này → trả Unknown thay vì đoán sai.
# Quan trọng khi có người thân/bạn bè cùng hộ — embedding gần nhau hơn người lạ.
CONFUSION_MARGIN = 0.10
# IOU threshold để xác định 2 bbox là cùng 1 khuôn mặt (cache lookup + update)
IOU_CACHE_THRESHOLD = 0.4

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
MAX_ENCODINGS_PER_PERSON = 50      # Max encodings kept per person after clustering (tăng từ 30 → 50 để bao phủ nhiều góc/ánh sáng hơn)
CLUSTERING_THRESHOLD = 0.25        # Face distance below which two encodings are near-duplicates
                                   # Tăng từ 0.15 → 0.25: distance 0.15 quá chặt, loại bỏ ảnh cùng người ở điều kiện khác nhau.
                                   # 0.25 chỉ dedup ảnh thực sự trùng, giữ lại đa dạng góc/ánh sáng.
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

# --- Real-time recognition performance ---
# Re-run face_recognition encoding every N frames; reuse cached name otherwise.
# Higher = faster, but name updates lag when a different person enters the bbox.
# 5 is a good balance: ~166ms lag at 30fps, no visible flicker.
RECOGNITION_INTERVAL = 10  # tăng từ 5→10: giảm 50% lần gọi dlib encoding (~0.6s lag tại 17FPS)
# Resize frame to this width before YOLO inference.
# 416 thay vì 640: nhanh hơn ~1.5-2x, giữ đủ độ phân giải cho khuôn mặt gần-trung bình.
# Nếu bạn cần nhận diện khuôn mặt ở rất xa, đổi lại 640.
YOLO_INPUT_WIDTH = 416

# =============================================================================
# --- Smart Home Security Extension ---
# =============================================================================

# RecognitionStabilizer — chống nhiễu nhận diện
# Mỗi cache-miss (~mỗi RECOGNITION_INTERVAL frames) tính là 1 "vote".
# Known person cần ít vote hơn để bật đèn nhanh; Unknown cần nhiều hơn để tránh báo nhầm.
STABILIZER_WINDOW = 5            # Theo dõi N cache-miss gần nhất per face
STABILIZER_MIN_KNOWN = 2         # ≥ 2/5 votes → confirm known person (~1s tại 20fps)
STABILIZER_MIN_UNKNOWN = 4       # ≥ 4/5 votes → confirm Unknown (~4s, tránh báo nhầm)
STABILIZER_TRACK_TIMEOUT = 60    # Xóa face track nếu mất > 60 frames (~3s)

# Alert — cảnh báo người lạ
ALERT_COOLDOWN_SECONDS = 30.0    # Thời gian chờ tối thiểu giữa hai lần alert (tránh spam)
ALERT_SNAPSHOT_DIR = os.path.join(_WORKSPACE_DIR, "alerts")  # Thư mục lưu ảnh người lạ
ALERT_ENABLE_SOUND = True        # Phát tiếng beep hệ thống khi có người lạ

# Entry log — lịch sử vào nhà
ENTRY_LOG_PATH = os.path.join(_WORKSPACE_DIR, "logs", "entry_log.jsonl")

# Device dispatch — kích hoạt thiết bị khi nhận diện thành viên
DISPATCH_COOLDOWN_SECONDS = 10.0  # Không dispatch lại cùng người trong vòng N giây

# MQTT broker (dùng env vars — không hardcode credentials vào code)
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "home/faces")
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", None)
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", None)

# Telegram Bot (optional — để trống nếu không dùng)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", None)

# Serial / Arduino (optional)
SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD_RATE = int(os.environ.get("SERIAL_BAUD_RATE", "9600"))

# --- Motion gate (CPU optimization for 24/7 operation) ---
# Kết hợp absdiff + MOG2 để quyết định có chạy YOLO hay không.
# Tất cả tính trên ảnh grayscale 160×90 (~0.6ms tổng) thay vì chạy YOLO mọi frame (~50-200ms).
#
# absdiff: mean pixel diff (0–255) giữa frame hiện tại và frame trước.
#   < threshold → không có gì thay đổi → skip YOLO ngay lập tức.
MOTION_ABSDIFF_THRESHOLD = 8
# MOG2: % pixels được đánh dấu foreground (0–100) sau khi absdiff trigger.
#   < threshold → chỉ là thay đổi ánh sáng, không phải người → skip YOLO.
MOTION_MOG2_THRESHOLD = 5
# Idle fallback: nếu YOLO không chạy quá N giây, force-run dù không có motion.
#   Đảm bảo người đứng yên lâu không bị MOG2 "nuốt" vào background.
MOTION_MAX_IDLE_SEC = 10

# --- State machine IDLE/ACTIVE (MotionGuard) ---
# IDLE: chỉ check motion mỗi N frame → ~5fps tại camera 30fps (~2% CPU)
IDLE_SAMPLE_EVERY = 6
# ACTIVE → IDLE: sau bao nhiêu frame liên tiếp YOLO không thấy mặt nào
# 150 frames ≈ 5 giây tại 30fps
IDLE_NO_FACE_FRAMES = 150
