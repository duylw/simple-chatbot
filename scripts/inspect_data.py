#!/usr/bin/env python3
"""
CLI Data Inspector for Temporal RAG QA System.
Allows fast inspection of CSV files, database videos, chunks, and ChromaDB embeddings directly in the terminal.
"""
import os
import csv
import sys
import pickle
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = PROJECT_ROOT / "data"

def inspect_csv_data():
    print("=" * 60)
    print("📊 1. KIỂM TRA DỮ LIỆU CSV (data/)")
    print("=" * 60)

    # 1. Videos CSV
    videos_file = DATA_DIR / "videos.csv"
    if videos_file.exists():
        with open(videos_file, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            print(f"🎬 Tổng số Videos trong {videos_file.name}: {len(reader)}")
            print("--- Top 3 videos mẫu ---")
            for i, row in enumerate(reader[:3], 1):
                print(f"  [{i}] ID: {row.get('video_uuid')} | Tên: {row.get('video_name')}")
    else:
        print(f"❌ Không tìm thấy file: {videos_file}")

    print()

    # 2. Chunks CSV
    chunks_file = DATA_DIR / "chunks.csv"
    if chunks_file.exists():
        with open(chunks_file, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            print(f"📄 Tổng số Chunks trong {chunks_file.name}: {len(reader)}")
            print("--- Top 2 chunks mẫu ---")
            for i, row in enumerate(reader[:2], 1):
                snippet = (row.get('content', '')[:100] + '...') if len(row.get('content', '')) > 100 else row.get('content', '')
                print(f"  [{i}] ID: {row.get('chunk_uuid')} | Timestamp: {row.get('timestamp')}s | Duration: {row.get('duration')}s")
                print(f"      Nội dung: {snippet}")
    else:
        print(f"❌ Không tìm thấy file: {chunks_file}")

    print()

    # 3. Vector Seed Pickle
    pkl_file = DATA_DIR / "vector_data_export.pkl"
    if pkl_file.exists():
        size_mb = pkl_file.stat().st_size / (1024 * 1024)
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                num_embeddings = len(data.get("ids", []))
                print(f"🧠 Vector Export ({pkl_file.name}): {size_mb:.2f} MB | {num_embeddings} pre-computed embeddings")
        except Exception as e:
            print(f"⚠️ Đọc file pickle thất bại: {e}")
    else:
        print(f"❌ Không tìm thấy file: {pkl_file}")

    print()
    print("=" * 60)
    print("🌐 2. TRUY CẬP DATA DASHBOARD TRÊN TRÌNH DUYỆT:")
    print("   Khi chạy Backend, truy cập: http://localhost:8000/dashboard")
    print("   Gradio Q&A Interface:      http://localhost:7860")
    print("=" * 60)

if __name__ == "__main__":
    inspect_csv_data()
