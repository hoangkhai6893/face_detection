#!/usr/bin/env python3
"""
Smart Home Entry Point — wires FaceRecognizer with security + smart-home components.

Usage:
    python src/main.py                          # full stack
    python src/main.py --no-alerts              # disable unknown-face alerts
    python src/main.py --no-dispatch            # disable device activation
    python src/main.py --no-logging             # disable entry log
    python src/main.py --verbose                # debug logging

Environment variables (secrets — do NOT hardcode):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID       # Telegram alerts
    MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, MQTT_TOPIC_PREFIX
    SERIAL_PORT, SERIAL_BAUD_RATE

All face-recognition params (--tolerance, --camera, --model, etc.) are the
same as recognizer.py.
"""

import argparse
import logging
import sys
from typing import Optional

import numpy as np

import config
from core.recognizer import FaceRecognizer, setup_logging
from services.alert_manager import AlertManager
from services.event_logger import EventLogger
from services.device_dispatcher import DeviceDispatcher
from core.recognition_stabilizer import RecognitionStabilizer
from services.notification_worker import NotificationWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Home Face Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Face recognizer args (mirrors recognizer.py) ---
    parser.add_argument("--dataset", "-d", default=config.DATASET_PATH)
    parser.add_argument("--model", "-m", default=config.MODEL_PATH)
    parser.add_argument("--tolerance", "-t", type=float, default=config.TOLERANCE)
    parser.add_argument("--detection-confidence", type=float, default=config.DETECTION_CONFIDENCE)
    parser.add_argument("--encoding-confidence", type=float, default=config.ENCODING_CONFIDENCE)
    parser.add_argument("--padding", "-p", type=int, default=config.RECOGNITION_PADDING)
    parser.add_argument("--encodings-file", default=None)
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    parser.add_argument("--recognition-interval", type=int, default=config.RECOGNITION_INTERVAL)
    parser.add_argument("--yolo-input-width", type=int, default=config.YOLO_INPUT_WIDTH)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=config.HEADLESS,
        help="Tắt cv2.imshow (bắt buộc trên Pi headless — hoặc set HEADLESS=true trong .env)",
    )
    parser.add_argument(
        "--active-process-every",
        type=int,
        default=config.ACTIVE_PROCESS_EVERY,
        help="Chạy full pipeline mỗi N frame khi ACTIVE (NUC: 1, Pi5: 2-3)",
    )
    parser.add_argument(
        "--face-backend",
        default=config.FACE_BACKEND,
        choices=["dlib", "sface"],
        help="Backend: 'dlib' (YOLO+face_recognition) hoặc 'sface' (YuNet+SFace)",
    )

    # --- Smart home extension args ---
    parser.add_argument("--no-alerts", action="store_true", help="Tắt cảnh báo người lạ")
    parser.add_argument("--no-dispatch", action="store_true", help="Tắt điều khiển thiết bị")
    parser.add_argument("--no-logging", action="store_true", help="Tắt entry log")
    parser.add_argument(
        "--alert-cooldown", type=float, default=config.ALERT_COOLDOWN_SECONDS,
        help="Thời gian chờ tối thiểu giữa hai lần alert (giây)",
    )
    parser.add_argument(
        "--dispatch-cooldown", type=float, default=config.DISPATCH_COOLDOWN_SECONDS,
        help="Thời gian chờ tối thiểu giữa hai lần dispatch cùng người (giây)",
    )
    parser.add_argument(
        "--stabilizer-window", type=int, default=config.STABILIZER_WINDOW,
        help="Số cache-miss theo dõi per face để chống nhiễu",
    )
    parser.add_argument(
        "--min-known", type=int, default=config.STABILIZER_MIN_KNOWN,
        help="Số votes tối thiểu để confirm known person",
    )
    parser.add_argument(
        "--min-unknown", type=int, default=config.STABILIZER_MIN_UNKNOWN,
        help="Số votes tối thiểu để confirm Unknown (báo động)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.verbose)

    logger.info("=== Smart Home Face Recognition ===")

    # --- Build components ---
    alert_manager: Optional[AlertManager] = None
    if not args.no_alerts:
        alert_manager = AlertManager(
            cooldown_seconds=args.alert_cooldown,
            logger=logger,
        )
        logger.info("AlertManager: ON (cooldown=%.0fs)", args.alert_cooldown)
    else:
        logger.info("AlertManager: OFF")

    event_logger: Optional[EventLogger] = None
    if not args.no_logging:
        event_logger = EventLogger(logger=logger)
        logger.info("EventLogger: ON → %s", config.ENTRY_LOG_PATH)
    else:
        logger.info("EventLogger: OFF")

    dispatcher: Optional[DeviceDispatcher] = None
    if not args.no_dispatch:
        dispatcher = DeviceDispatcher(
            dataset_path=args.dataset,
            default_cooldown=args.dispatch_cooldown,
            logger=logger,
        )
        logger.info("DeviceDispatcher: ON (cooldown=%.0fs)", args.dispatch_cooldown)
    else:
        logger.info("DeviceDispatcher: OFF")

    stabilizer = RecognitionStabilizer(
        window_size=args.stabilizer_window,
        min_count_known=args.min_known,
        min_count_unknown=args.min_unknown,
    )
    logger.info(
        "RecognitionStabilizer: window=%d, min_known=%d, min_unknown=%d",
        args.stabilizer_window, args.min_known, args.min_unknown,
    )

    # NotificationWorker: xử lý Telegram / MQTT / disk I/O trên background thread
    # → camera loop không bao giờ bị block bởi network/disk
    worker = NotificationWorker(
        alert_manager=alert_manager,
        event_logger=event_logger,
        dispatcher=dispatcher,
        logger=logger,
    )

    # Frame counter shared with callback (closure)
    frame_state = {"count": 0}

    def on_recognition_update(raw_name: str, bbox: tuple, frame: np.ndarray) -> None:
        frame_state["count"] += 1
        # stabilizer.update() chạy đồng bộ: chỉ RAM ops, < 0.1ms
        event = stabilizer.update(raw_name, bbox, frame_state["count"], frame)
        if event is not None:
            # Đẩy vào queue — non-blocking (< 1ms)
            # I/O thực sự (Telegram, MQTT, disk) chạy trên NotificationWorker thread
            worker.submit(event)

    # --- Build FaceRecognizer with callback ---
    try:
        recognizer = FaceRecognizer(
            dataset_path=args.dataset,
            model_path=args.model,
            tolerance=args.tolerance,
            detection_confidence=args.detection_confidence,
            encoding_confidence=args.encoding_confidence,
            padding=args.padding,
            encodings_file=args.encodings_file,
            recognition_interval=args.recognition_interval,
            yolo_input_width=args.yolo_input_width,
            headless=args.headless,
            active_process_every=args.active_process_every,
            face_backend=args.face_backend,
            logger=logger,
            on_recognition_update=on_recognition_update,
        )
    except FileNotFoundError as e:
        logger.error("Không tìm thấy file: %s", e)
        sys.exit(1)

    # --- Run ---
    try:
        recognizer.run_recognition(
            camera_id=args.camera,
            frame_width=args.width,
            frame_height=args.height,
        )
    finally:
        worker.shutdown()
        if dispatcher:
            dispatcher.shutdown()
        logger.info("Smart home system stopped.")


if __name__ == "__main__":
    main()
