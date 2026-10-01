import cv2
from pathlib import Path
from collections import Counter
import pandas as pd

# 調べるフォルダ
SEARCH_DIRS = [
    Path("test_videos"),
    Path("mlb-yt-dataset"),
]

video_files = []

for search_dir in SEARCH_DIRS:
    if search_dir.exists():
        video_files.extend(search_dir.rglob("*.mp4"))

print("動画数:", len(video_files))

records = []

for video_path in video_files:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    records.append(
        {
            "video": video_path.name,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
        }
    )

df = pd.DataFrame(records)

# 解像度
resolution_counts = df.groupby(["width", "height"]).size().sort_values(ascending=False)

# FPS
fps_counts = df["fps"].value_counts().sort_index()

print()
print("=" * 50)
print("解像度ごとの動画数")
print("=" * 50)

for (width, height), count in resolution_counts.items():
    print(f"{width} x {height} : {count}本")

print()
print("=" * 50)
print("FPSごとの動画数")
print("=" * 50)

for fps, count in fps_counts.items():
    print(f"{fps} fps : {count}本")

# CSV保存
output_path = Path("output/video_info.csv")
output_path.parent.mkdir(exist_ok=True)

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print()
print("詳細データ保存:", output_path)
