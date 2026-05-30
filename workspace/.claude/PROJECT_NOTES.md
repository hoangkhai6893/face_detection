# Family Face Recognition System — Project Notes

## Tổng quan
Hệ thống nhận diện khuôn mặt cho gia đình, kết hợp:
- **YOLO v11 Nano** (`src/yolo/yolov11n-face.pt`) — phát hiện khuôn mặt trong frame
- **dlib ResNet** (qua thư viện `face_recognition`) — tạo vector embedding 128 chiều
- **KHÔNG train neural network từ đầu** — chỉ thu thập ảnh → extract embedding → lưu pkl

## Môi trường chạy
- **Python**: `python3` (3.13) — **KHÔNG dùng `python3.11`** (không còn tồn tại)
- **Packages cần thiết**: `face_recognition`, `ultralytics`, `cv2`, `onnxruntime`, `scikit-image`, `questionary`, `pytest`
- **Cài dlib từ source**: cần `g++-12`
  ```bash
  sudo apt-get install g++-12
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
  pip3 install dlib face_recognition
  ```

## Cấu trúc thư mục quan trọng
```
workspace/
├── src/
│   ├── config.py              ← Tất cả thông số cấu hình
│   ├── recognizer.py          ← Pipeline nhận diện real-time
│   ├── training_manager.py    ← CLI thu thập data + quản lý người
│   ├── augment_dataset.py     ← Tạo biến thể ảnh (×8 per image)
│   ├── optimize_encodings.py  ← Cluster encodings, giữ max 30/người
│   ├── core/
│   │   ├── encoder.py         ← rebuild_encodings() + update_person_encodings()
│   │   ├── frame_extractor.py ← Quality/diversity filter cho training frames
│   │   └── face_utils.py      ← Crop khuôn mặt từ bbox
│   └── yolo/yolov11n-face.pt  ← YOLO model (hoặc .onnx nếu đã export)
├── model/
│   ├── face_encodings_hybrid.pkl         ← Encodings đang dùng (30/người)
│   └── face_encodings_hybrid_backup.pkl  ← Backup tự động trước mỗi lần update
├── family_images/
│   ├── Khai/    ← 3,998 gốc + 31,984 augmented
│   └── MaiAnh/  ← 406 gốc + 3,248 augmented
└── tests/       ← 93 tests (pytest)
```

## Thông số nhận diện (config.py) — đã tinh chỉnh 2026-05-30
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `TOLERANCE` | **0.5** | Giảm từ 0.6 — chặt hơn, ít nhầm giữa bạn bè/người thân |
| `RECOGNITION_TOP_K` | **5** | Tăng từ 3 — voting chính xác hơn |
| `CONFUSION_MARGIN` | **0.10** | **Mới** — nếu người thứ 2 cách winner < 0.10 → trả "Unknown" |
| `MAX_ENCODINGS_PER_PERSON` | 30 | Sau optimize |
| `CLUSTERING_THRESHOLD` | 0.15 | Khoảng cách để gộp encodings giống nhau |

## Cải tiến đã thực hiện (2026-05-30)

### 1. Confidence Margin Guard (recognizer.py)
**Vấn đề**: Hệ thống nhầm giữa Khai và bạn (MaiAnh) khi distance gần nhau.
**Giải pháp**: Sau khi chọn winner, check khoảng cách người thứ 2 trên **toàn bộ** encodings:
- Nếu `distance_2nd - distance_winner < CONFUSION_MARGIN (0.10)` → trả "Unknown"
- Nằm trong hàm `recognize_face_in_region()` (~dòng 372)

### 2. Incremental Update (core/encoder.py)
**Vấn đề**: Mỗi lần thêm data mới phải rebuild toàn bộ 39K ảnh (~4 giờ).
**Giải pháp**: Hàm `update_person_encodings()` — chỉ encode ảnh mới, giữ nguyên người khác:
```
Trước: pkl = [Khai×30, MaiAnh×30]
Thêm 100 ảnh mới cho MaiAnh (10 giây):
  → Encode 100 ảnh mới + merge với 30 cũ → cluster → 30 đại diện tốt nhất
Sau:  pkl = [Khai×30, MaiAnh×30 (cập nhật)]
```
**DataCollectionSession** tự động dùng incremental update sau mỗi lần thu thập.

### 3. Rebuild thủ công vẫn còn
Menu "Rebuild encodings (option 4)" trong `training_manager.py` vẫn full rebuild — dùng khi muốn reset hoàn toàn.

## Pipeline "đào tạo" đúng nghĩa
```
1. Thu thập ảnh   → python3 src/training_manager.py  → menu: Thu thập dữ liệu mới
2. Augment        → menu: Augment dataset             → tạo 8 biến thể/ảnh
3. Update/Rebuild → tự động sau thu thập (incremental) HOẶC menu: Rebuild (full)
4. Optimize       → menu: Optimize encodings          → cluster → 30/người (KHÔNG cần thiết nếu dùng incremental)
```

## Lệnh hay dùng
```bash
# Nhận diện real-time
python3 src/recognizer.py --camera 0

# Thu thập data + quản lý người
python3 src/training_manager.py

# Chạy toàn bộ tests
python3 -m pytest tests/ -v

# Smoke test nhanh (không cần camera)
python3 .claude/skills/run-workspace/driver.py

# Export YOLO sang ONNX (nhanh hơn 1.5-2x)
cd src && python3 export_onnx.py
```

## Dataset hiện tại
| Người | Ảnh gốc | Augmented | Encodings trong pkl |
|-------|---------|-----------|---------------------|
| Khai | 3,998 | 31,984 | 30 |
| MaiAnh | 406 | 3,248 | 30 |

**Vấn đề còn lại**: MaiAnh cần thêm 2–3 session quay từ điều kiện khác nhau (ánh sáng, góc mặt) để tăng độ chính xác phân biệt với Khai.

## Gotchas quan trọng
- `python3.11` **không tồn tại** trong môi trường hiện tại — dùng `python3`
- `PersonManager._rebuild()` ghi vào `config.ENCODINGS_DIR` thật → luôn dùng `rebuild=False` trong test
- `CONFUSION_MARGIN` có thể cần tinh chỉnh: nếu quá nhiều "Unknown" → tăng lên 0.15; nếu vẫn nhầm → giảm xuống 0.08
- Backup tự động mỗi lần update pkl: `face_encodings_hybrid_backup.pkl`
- `optimize_encodings.py` xử lý lại TẤT CẢ ảnh từ file — cực chậm với 39K ảnh, **không nên chạy lại**; dùng incremental thay thế
