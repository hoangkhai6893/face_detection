# Phase 1 — Pi-Ready & 24/7 Reliability

> **Mục tiêu:** Làm hệ thống chạy được trên Pi 5 headless 24/7 với stack YOLO+dlib hiện tại.
> **Không đổi:** model, encodings, accuracy logic, services layer.
> **Rủi ro:** THẤP — không thay model, không thay algorithm. Rollback = `git revert`.

---

## 1.1 Headless Mode

**Vấn đề:** `recognizer.py:568` gọi `cv2.imshow()` vô điều kiện → exception trên Pi headless (không có X server / dùng `opencv-python-headless`).

### Config thêm vào `src/config.py`

```python
# --- Headless mode ---
# True = tắt toàn bộ cv2 display. Bắt buộc khi deploy Pi headless.
HEADLESS: bool = os.environ.get("HEADLESS", "false").lower() == "true"
```

### Thay đổi `src/main.py`

Thêm argument sau block `--verbose`:
```python
parser.add_argument(
    "--headless",
    action="store_true",
    default=config.HEADLESS,
    help="Tắt display (bắt buộc trên Pi headless)",
)
```

Truyền xuống khi init + run:
```python
recognizer = FaceRecognizer(
    ...
    headless=args.headless,
)
recognizer.run_recognition(
    camera_id=args.camera,
    frame_width=args.width,
    frame_height=args.height,
    headless=args.headless,  # pass qua để guard bên trong
)
```

### Thay đổi `src/core/recognizer.py`

**`__init__` — thêm param:**
```python
def __init__(
    self,
    ...
    headless: bool = False,
):
    ...
    self.headless = headless
```

**`run_recognition()` — thêm param + guard display:**
```python
def run_recognition(self, camera_id=0, frame_width=1280, frame_height=720, headless=False):
    ...
    # Dòng 546 — draw results
    if face_detections and recognition_results:
        if not headless:
            frame = self.draw_results(frame, face_detections, recognition_results)

    # Dòng 563-566 — info overlay (FPS, state, faces)
    if not headless:
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, ...)

    # Dòng 568 — imshow
    if not headless:
        cv2.imshow('Face Recognition', frame)

    # Dòng 570-572 — waitKey + 'q' quit
    if not headless:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
```

**`finally` block (dòng 577-580):**
```python
finally:
    stop_capture.set()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    self.logger.info("Recognition system stopped")
```

### `.env.example` thêm:
```bash
# Headless mode (Pi deploy)
HEADLESS=true
```

---

## 1.2 ACTIVE Process Throttle

**Vấn đề:** khi ACTIVE, YOLO + dlib chạy 100% mỗi frame. Pi5 cần ~60-80ms/frame → bottleneck.
**Giải pháp:** chạy full pipeline mỗi N frame; frame còn lại tái dùng kết quả cuối.

### Config thêm vào `src/config.py`

```python
# --- ACTIVE state throttle ---
# Chỉ chạy full pipeline (YOLO + dlib) mỗi N frame khi ACTIVE.
# N=1 = không throttle (giữ hành vi cũ). N=2 = chạy 1/2 frame.
# Pi5 khuyến nghị: 2. Tăng lên 3 nếu vẫn nóng.
ACTIVE_PROCESS_EVERY: int = int(os.environ.get("ACTIVE_PROCESS_EVERY", "1"))
```

### Thay đổi `src/core/recognizer.py`

**`__init__` — thêm:**
```python
self.active_process_every = active_process_every  # tham số mới từ config
self._active_skip_count: int = 0                  # counter frame throttle
```

**`run_recognition()` — trong block `if _should_run:`:**

Hiện tại (dòng 494-543):
```python
if _should_run:
    # YOLO detect ...
    # recognize per face ...
    _last_face_detections = face_detections
    _last_recognition_results = recognition_results
```

Thay bằng:
```python
if _should_run:
    self._active_skip_count += 1
    _do_full_pipeline = (
        self._motion_guard.state.name == "IDLE"  # probe burst: luôn chạy
        or self._active_skip_count >= self.active_process_every
    )

    if _do_full_pipeline:
        self._active_skip_count = 0
        # --- giữ nguyên toàn bộ code YOLO detect + recognition ở đây ---
        # YOLO resize + detect_faces_yolo()
        # motion_guard.report_faces()
        # per-face IOU cache + recognize
        # _last_face_detections = face_detections
        # _last_recognition_results = recognition_results
    else:
        # Throttled frame: skip YOLO, tái dùng kết quả + báo motion guard
        face_detections = _last_face_detections
        recognition_results = _last_recognition_results
        # Vẫn báo motion guard để đếm "no-face" frames chính xác
        self._motion_guard.report_faces(len(face_detections))
```

**Main.py — thêm argument:**
```python
parser.add_argument(
    "--active-process-every",
    type=int,
    default=config.ACTIVE_PROCESS_EVERY,
    help="Chạy full pipeline mỗi N frame khi ACTIVE (Pi5 khuyến nghị 2)",
)
```

Truyền vào FaceRecognizer:
```python
recognizer = FaceRecognizer(
    ...
    active_process_every=args.active_process_every,
)
```

---

## 1.3 Camera Resolution cho Pi

**Vấn đề:** mặc định 1280×720 quá lớn cho Pi — memory bandwidth + YOLO downscale overhead.

### Config `src/config.py` — đổi default:

```python
# --- Camera / display ---
# Pi deploy: dùng .env FRAME_WIDTH=640 FRAME_HEIGHT=480
# NUC dev: giữ 1280×720 (hoặc set trong .env riêng)
FRAME_WIDTH  = int(os.environ.get("FRAME_WIDTH",  "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "720"))
```

### `.env` Pi5:
```bash
FRAME_WIDTH=640
FRAME_HEIGHT=480
YOLO_INPUT_WIDTH=416   # Pi5 kham; thử 320 nếu cần nhanh hơn
```

Không cần sửa code — `run_recognition()` đã nhận `frame_width/frame_height` từ `main.py` qua `args.width/args.height`, và `config.FRAME_WIDTH/HEIGHT` đã là default.

---

## 1.4 HOG Fallback Logging

**Vấn đề:** `_fallback_extract_encoding()` (recognizer.py:241) chạy HOG full-image (~1s+ trên Pi) mà không log gì → đột ngột giật không debug được.

### Thay đổi `src/core/recognizer.py` — `_fallback_extract_encoding()`:

```python
def _fallback_extract_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
    """Fallback: HOG full-image scan khi YOLO miss. Chậm (~500ms-1s trên Pi)."""
    t0 = time.time()
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        elapsed_ms = (time.time() - t0) * 1000
        if face_encodings:
            self.logger.warning(
                "HOG fallback triggered (YOLO missed) — elapsed %.0fms", elapsed_ms
            )
            return face_encodings[0]
        return None
    except Exception:
        return None
```

Thêm counter để track trong health log:
```python
# __init__:
self._hog_fallback_count: int = 0

# _fallback_extract_encoding, sau warning:
self._hog_fallback_count += 1
```

---

## 1.5 systemd Service (Pi 5)

**File:** `scripts/install_service.sh` (NEW)

```bash
#!/usr/bin/env bash
# Cài đặt face-recognition như systemd service trên Pi 5.
# Chạy: sudo bash scripts/install_service.sh [working_dir] [user]
set -e

WORKDIR="${1:-$(pwd)}"
SERVICE_USER="${2:-$USER}"
VENV_PYTHON="$WORKDIR/venv/bin/python"
SERVICE_NAME="face-recognition"

cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=Family Face Recognition System
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${WORKDIR}
EnvironmentFile=${WORKDIR}/.env
ExecStart=${VENV_PYTHON} src/main.py --headless
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
# Tắt gracefully khi SIGTERM (KboardInterrupt handler đã có)
KillSignal=SIGTERM
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl start ${SERVICE_NAME}
echo "✓ Service ${SERVICE_NAME} installed và started."
echo "  Xem log: journalctl -fu ${SERVICE_NAME}"
echo "  Dừng:    sudo systemctl stop ${SERVICE_NAME}"
```

**Lệnh quản lý sau cài đặt:**
```bash
sudo systemctl status face-recognition    # xem trạng thái
journalctl -fu face-recognition           # stream log real-time
sudo systemctl restart face-recognition   # restart thủ công
sudo systemctl stop face-recognition      # dừng
```

---

## 1.6 EventLogger — Log Rotation Theo Ngày

**Vấn đề:** `entry_log.jsonl` không giới hạn → file phình + `read_recent()` O(N) → nguy cơ mòn SD card.

### Thay đổi `src/services/event_logger.py`

**Thay đổi `__init__`:**
```python
def __init__(self, log_path: str = config.ENTRY_LOG_PATH) -> None:
    self._base_path = Path(log_path)         # base path (không đổi)
    self._base_path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()
    self._cleanup_old_logs()                 # xóa file > 30 ngày khi khởi động
```

**Thêm method `_today_path()`:**
```python
def _today_path(self) -> Path:
    """Trả về path file log cho ngày hôm nay."""
    stem = self._base_path.stem   # "entry_log"
    date = datetime.now().strftime("%Y%m%d")
    return self._base_path.parent / f"{stem}_{date}.jsonl"
```

**Cập nhật `_append_line()`:**
```python
def _append_line(self, event: RecognitionEvent) -> None:
    with self._lock:
        path = self._today_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(event), ensure_ascii=False) + "\n")
```

**Cập nhật `read_recent(n=50)`:**
```python
def read_recent(self, n: int = 50) -> List[dict]:
    """Đọc N dòng gần nhất từ file hôm nay."""
    try:
        lines = self._today_path().read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-n:]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []
```

**Thêm `_cleanup_old_logs()`:**
```python
def _cleanup_old_logs(self, keep_days: int = 30) -> None:
    """Xóa file log cũ hơn keep_days ngày."""
    cutoff = datetime.now().timestamp() - keep_days * 86400
    stem = self._base_path.stem
    for f in self._base_path.parent.glob(f"{stem}_*.jsonl"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            logging.getLogger(__name__).info("EventLogger: deleted old log %s", f.name)
```

---

## 1.7 Health Log Định Kỳ

Thêm vào `src/main.py` — sau khi khởi động, chạy thread báo cáo health mỗi giờ:

```python
def _start_health_logger(recognizer: FaceRecognizer, interval_sec: int = 3600) -> None:
    """Log health metrics mỗi interval_sec. Daemon thread — tự chết khi main exit."""
    import threading
    logger = logging.getLogger("health")

    def _run():
        while True:
            time.sleep(interval_sec)
            motion = recognizer._motion_guard
            avg_fps = (1.0 / np.mean(recognizer.detection_times)
                       if recognizer.detection_times else 0.0)
            logger.info(
                "HEALTH | state=%s fps=%.1f faces_cached=%d hog_fallback=%d",
                motion.state.name,
                avg_fps,
                len(recognizer._face_cache),
                recognizer._hog_fallback_count,
            )
            recognizer._hog_fallback_count = 0  # reset counter sau mỗi report

    t = threading.Thread(target=_run, daemon=True, name="HealthLogger")
    t.start()
```

Gọi sau khi init components, trước `recognizer.run_recognition()`.

---

## Tiêu Chí Kiểm Tra Phase 1

```bash
# 1. Headless không crash
HEADLESS=true python src/main.py --headless
# Kỳ vọng: khởi động bình thường, không ImportError, không cv2 exception

# 2. Test suite vẫn pass
pytest tests/ -v
# Kỳ vọng: 93/93 pass

# 3. ACTIVE throttle không mất detection
python src/main.py --active-process-every 2 --verbose
# Đứng trước camera, đi lại. Kỳ vọng: nhận diện đúng, không "Unknown" giả nhiều hơn.
# Chấp nhận: lag nhận tên tăng ~100ms (1 frame thêm @ 15fps)

# 4. systemd service
sudo bash scripts/install_service.sh
sudo systemctl status face-recognition   # ← active (running)
# Rút USB camera → systemctl status sau 10s: restarting/active

# 5. Log rotation
# Chạy 2 ngày → kiểm tra logs/entry_log_YYYYMMDD.jsonl tồn tại mỗi ngày
# File ngày > 30 ngày: tự xóa sau khi restart service

# 6. HOG fallback log
# Giảm DETECTION_CONFIDENCE=0.9 (bắt YOLO miss nhiều)
# Kỳ vọng: "WARNING: HOG fallback triggered" xuất hiện trong log
```

---

## Rollback Phase 1

```bash
git diff --stat           # xem tất cả file đã đổi
git stash                 # hoặc git revert <commit>
sudo systemctl stop face-recognition
sudo systemctl disable face-recognition
sudo rm /etc/systemd/system/face-recognition.service
```

---

## .env Hoàn Chỉnh cho Pi 5

```bash
# Pi 5 production .env
HEADLESS=true
FRAME_WIDTH=640
FRAME_HEIGHT=480
YOLO_INPUT_WIDTH=416
ACTIVE_PROCESS_EVERY=2

# Motion tuning (optional — đo trên Pi thật rồi điều chỉnh)
MOTION_MAX_IDLE_SEC=60
IDLE_PROBE_BURST=5

# Smart home (cấu hình theo môi trường)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
MQTT_HOST=192.168.1.xxx
MQTT_PORT=1883
```
