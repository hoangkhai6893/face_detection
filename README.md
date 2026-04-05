# Family Face Recognition System

Hệ thống nhận diện khuôn mặt thời gian thực dành cho gia đình, kết hợp **YOLO** (YOLOv11/v12) để phát hiện khuôn mặt và **face_recognition** (dlib) để nhận diện danh tính. Hệ thống hỗ trợ thu thập dữ liệu thông minh, tăng cường dữ liệu (augmentation), tối ưu hóa encoding và triển khai qua Docker.

---

## Tổng quan

| Thành phần | Công nghệ |
|---|---|
| Phát hiện khuôn mặt | YOLOv11n / YOLOv11s / YOLOv12s |
| Nhận diện danh tính | face_recognition (dlib, 128-d vector) |
| Thu thập dữ liệu | Lọc chất lượng (sharpness, brightness) + lọc đa dạng (SSIM) |
| Tăng cường dữ liệu | 8 biến thể: blur, motion blur, low-light, noise, JPEG nén |
| Tối ưu encoding | Clustering để loại bỏ ~90% encoding trùng lặp |
| Triển khai | Docker + Dev Container (VS Code) |

### Luồng xử lý chính

```
Camera/Video
    ↓
YOLO phát hiện khuôn mặt (confidence > 0.3)
    ↓
Cắt vùng khuôn mặt (padding 15px)
    ↓
face_recognition → vector 128 chiều
    ↓
So sánh khoảng cách với database (tolerance 0.6)
    ↓
Hiển thị tên + độ tự tin lên frame
```

### Cấu trúc thư mục

```
face_detection/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
│
├── .devcontainer/
│   └── devcontainer.json          # Cấu hình VS Code Dev Container
│
└── workspace/
    ├── src/
    │   ├── config.py              # Tất cả tham số cấu hình tập trung
    │   ├── recognizer.py          # Nhận diện thời gian thực
    │   ├── training_manager.py    # Thu thập & quản lý dữ liệu (CLI)
    │   ├── augment_dataset.py     # Tăng cường dữ liệu ảnh
    │   ├── optimize_encodings.py  # Tối ưu encoding qua clustering
    │   ├── benchmark_models.py    # So sánh hiệu năng các model YOLO
    │   └── core/
    │       ├── face_utils.py      # Trích xuất vùng khuôn mặt
    │       ├── encoder.py         # Xây dựng encoding database
    │       └── frame_extractor.py # Lọc chất lượng & đa dạng frame
    │
    ├── family_images/             # Dataset ảnh huấn luyện
    ├── model/                     # File encoding (.pkl)
    ├── data/                      # Video thô (.mp4)
    └── tests/                     # 37 pytest test cases
```

---

## Yêu cầu hệ thống

- Docker & Docker Compose
- Webcam hoặc video file
- X11 (để hiển thị GUI khi dùng Docker trên Linux)
- VS Code + extension **Dev Containers** (nếu dùng Dev Container)

---

## Cài đặt & Chạy

### 1. Clone repository

```bash
git clone <repo-url>
cd face_detection
```

### 2. Chạy trực tiếp (không Docker)

```bash
cd workspace
pip install -r ../requirements.txt

# Nhận diện thời gian thực
python src/recognizer.py

# Thu thập dữ liệu / quản lý người dùng
python src/training_manager.py
```

### 3. Chạy với Docker

**Build image:**

```bash
docker compose build
```

**Cho phép container hiển thị GUI (X11):**

```bash
xhost +local:docker
```

**Khởi động container:**

```bash
docker compose up -d
```

**Truy cập vào container:**

```bash
docker exec -it face_detection bash
```

**Chạy nhận diện khuôn mặt trong container:**

```bash
# Trong container, workdir là /home/dkhai/workspace/src
python recognizer.py

# Tùy chọn:
python recognizer.py --camera 0 --tolerance 0.6
python recognizer.py --model yolo/yolov11n-face.pt    # nhanh hơn
python recognizer.py --verbose                         # log chi tiết
```

> Nhấn `q` để thoát chương trình nhận diện.

**Dừng container:**

```bash
docker compose down
```

---

## Dev Container (VS Code)

Dev Container cho phép bạn lập trình trực tiếp **bên trong môi trường container** mà không cần cài đặt dependencies trên máy host.

### Thiết lập

1. Cài extension **Dev Containers** trong VS Code (`ms-vscode-remote.remote-containers`)

2. Mở thư mục `face_detection` trong VS Code

3. Nhấn `F1` → gõ **"Dev Containers: Reopen in Container"** → Enter

   Hoặc click biểu tượng `><` góc dưới trái → **"Reopen in Container"**

4. VS Code sẽ tự động:
   - Build Docker image (nếu chưa có)
   - Khởi động container từ `docker-compose.yml`
   - Mount workspace vào `/home/dkhai/workspace`
   - Kết nối VS Code vào container với user `dkhai`

5. Mở terminal trong VS Code (`Ctrl+`` ` ``) — terminal này chạy **bên trong container**

### Cấu hình Dev Container

File [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json):

```json
{
    "name": "face_detection",
    "service": "face_detection",
    "dockerComposeFile": "../docker-compose.yml",
    "workspaceFolder": "/home/dkhai/workspace",
    "remoteUser": "dkhai"
}
```

- **service**: sử dụng service `face_detection` trong `docker-compose.yml`
- **workspaceFolder**: thư mục mặc định khi mở VS Code trong container
- **remoteUser**: user `dkhai` (non-root, có quyền sudo)

### Thoát Dev Container

Nhấn `F1` → **"Dev Containers: Reopen Folder Locally"** để quay lại môi trường host.

---

## Sử dụng chi tiết

### Nhận diện thời gian thực

```bash
python src/recognizer.py
python src/recognizer.py --camera 0                    # chọn camera
python src/recognizer.py --tolerance 0.5               # nghiêm ngặt hơn
python src/recognizer.py --detection-confidence 0.4    # ngưỡng YOLO
python src/recognizer.py --model yolo/yolov12s-face.pt # model chính xác hơn
```

| Model | Tốc độ | Độ chính xác |
|---|---|---|
| yolov11n-face.pt | ~17 FPS (nhanh nhất) | Trung bình |
| yolov11s-face.pt | ~12 FPS | Tốt |
| yolov12s-face.pt | ~7 FPS | Tốt nhất |

### Thu thập dữ liệu & quản lý người dùng

```bash
python src/training_manager.py
```

Menu tương tác gồm 7 lựa chọn:
1. Thu thập ảnh từ camera hoặc video
2. Quản lý danh sách người (thêm/xóa/reset)
3. Tăng cường dataset (augmentation)
4. Rebuild encoding database
5. Tối ưu encoding (clustering)
6. Điều chỉnh tham số trích xuất frame
7. Trợ giúp

### Tăng cường dữ liệu

```bash
python src/augment_dataset.py                  # tất cả mọi người
python src/augment_dataset.py --person Khai    # chỉ một người
```

Tạo 8 biến thể mỗi ảnh: blur nhẹ/nặng, motion blur, ánh sáng yếu, nhiễu, JPEG nén.

### Tối ưu encoding

```bash
python src/optimize_encodings.py
```

Giảm ~90% encoding trùng lặp bằng clustering, giữ nguyên độ chính xác nhận diện.

### Chạy tests

```bash
cd workspace
pytest tests/ -v                        # tất cả 37 tests
pytest tests/test_frame_quality.py -v   # kiểm tra bộ lọc chất lượng frame
pytest tests/test_person_manager.py -v  # kiểm tra CRUD người dùng
```

---

## Cấu hình

Tất cả tham số tập trung tại [workspace/src/config.py](workspace/src/config.py):

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `TOLERANCE` | 0.6 | Ngưỡng nhận diện (thấp hơn = chặt hơn) |
| `DETECTION_CONFIDENCE` | 0.3 | Ngưỡng YOLO phát hiện khuôn mặt |
| `FRAME_MIN_LAPLACIAN` | 15 | Độ sắc nét tối thiểu của frame |
| `FRAME_MIN_FACE_PX` | 70 | Kích thước khuôn mặt tối thiểu (px) |
| `MAX_ENCODINGS_PER_PERSON` | 30 | Số encoding tối đa sau tối ưu |
| `RECOGNITION_INTERVAL` | 5 | Re-encode sau mỗi N frame |

---

## Triển khai Docker — Chi tiết kỹ thuật

### Dockerfile

- Base image: `python:3.13-slim`
- Timezone: `Asia/Tokyo`
- System packages: `cmake`, `libgl1`, `python3-opencv`, Node.js 20
- Python packages: cài từ `requirements.txt` (fallback nếu dlib build lỗi)
- User: `dkhai` (UID 1000, non-root, có sudo không cần password)

### docker-compose.yml — Volumes quan trọng

| Volume | Mục đích |
|---|---|
| `./workspace:/home/dkhai/workspace` | Mount code vào container |
| `/tmp/.X11-unix:/tmp/.X11-unix` | Cho phép hiển thị GUI |
| `~/.Xauthority:/home/dkhai/.Xauthority` | X11 authentication |
| `/dev:/dev` | Truy cập camera và thiết bị |
| `/var/run/dbus:/var/run/dbus` | D-Bus cho phần cứng |

> Container chạy ở chế độ `privileged: true` và `network_mode: host` để hỗ trợ camera và X11 forwarding.

---

## Ghi chú

- Nếu camera không hiển thị GUI khi dùng Docker, chạy `xhost +local:docker` trên host trước.
- dlib có thể build lỗi trên một số hệ thống; Dockerfile đã có fallback tự động.
- Encoding database được lưu tại `workspace/model/face_encodings_hybrid.pkl`.
- Mỗi lần thêm người mới hoặc thêm ảnh mới, cần **Rebuild encodings** qua menu `training_manager.py`.
