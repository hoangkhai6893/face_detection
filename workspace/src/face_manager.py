#!/usr/bin/env python3
"""
face_manager.py — Cập nhật dữ liệu khuôn mặt và train lại model.

Chạy: python src/face_manager.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.cli import main

if __name__ == "__main__":
    main()
