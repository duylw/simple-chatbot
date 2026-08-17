#!/usr/bin/env python3
"""
Script to automatically download lecture videos from Google Drive
and filter ONLY .mp4 video files into data/videos/ with NFC Unicode normalization.
"""

import os
import sys
import shutil
import unicodedata
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
TEMP_DOWNLOAD_DIR = PROJECT_ROOT / "data" / "_temp_gdrive_download"

GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1tueMJDG64fmEyqSL5eF3ABNRRIKkAhn4"


def download_and_filter_videos():
    try:
        import gdown
    except ImportError:
        print("'gdown' chưa được cài đặt. Hãy chạy: uv run --with gdown python scripts/download_videos.py")
        sys.exit(1)

    TARGET_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BẮT ĐẦU TẢI DỮ LIỆU TỪ GOOGLE DRIVE...")
    print(f"URL Thư mục: {GDRIVE_FOLDER_URL}")
    print(f"Thư mục đích: {TARGET_VIDEOS_DIR}")
    print("=" * 65)

    try:
        # Download folder from Google Drive into temporary directory
        print("\nĐang kết nối và tải các tệp từ Google Drive (gdown)...")
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=str(TEMP_DOWNLOAD_DIR),
            quiet=False,
            use_cookies=False,
        )

        print("\nĐang quét và lọc CHỈ các tệp video (.mp4)...")
        video_count = 0

        # Recursively search for all .mp4 files in the downloaded temp directory
        for file_path in TEMP_DOWNLOAD_DIR.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() == ".mp4":
                # Normalize unicode filename to NFC (chuẩn Unicode tiếng Việt)
                normalized_name = unicodedata.normalize("NFC", file_path.name)
                dest_path = TARGET_VIDEOS_DIR / normalized_name

                # Move or copy the video file
                shutil.move(str(file_path), str(dest_path))
                video_count += 1
                print(f"[Đã chuyển] {normalized_name} ({dest_path.stat().st_size / (1024 * 1024):.1f} MB)")

        print("\n" + "=" * 65)
        print(f"HOÀN TẤT! Đã lọc và lưu thành công {video_count} video vào '{TARGET_VIDEOS_DIR}'.")
        print("=" * 65)

    except Exception as e:
        print(f"\nCó lỗi xảy ra trong quá trình tải: {e}")
    finally:
        # Clean up temporary download directory
        if TEMP_DOWNLOAD_DIR.exists():
            shutil.rmtree(TEMP_DOWNLOAD_DIR, ignore_errors=True)
            print("Đã dọn dẹp các thư mục rác / tạm thời.")


if __name__ == "__main__":
    download_and_filter_videos()
