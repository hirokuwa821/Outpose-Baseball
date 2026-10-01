from pathlib import Path
import hashlib
import cv2
import pandas as pd

VIDEO_DIR = Path("test_videos")
OUTPUT_FILE = Path("output/video_duplicate_check.csv")


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video": video_path.name,
            "size_bytes": video_path.stat().st_size,
            "frames": None,
            "fps": None,
            "duration": None,
            "width": None,
            "height": None,
            "hash": None,
        }

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frames / fps if fps > 0 else None

    cap.release()

    # ファイル全体のSHA-256
    sha256 = hashlib.sha256()

    with open(video_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return {
        "video": video_path.name,
        "size_bytes": video_path.stat().st_size,
        "frames": frames,
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "hash": sha256.hexdigest(),
    }


# 動画一覧
videos = sorted(
    [
        p
        for p in VIDEO_DIR.iterdir()
        if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]
    ]
)

print(f"動画数: {len(videos)}")
print()

results = []

for i, video in enumerate(videos, start=1):
    print(f"[{i}/{len(videos)}] 調査中: {video.name}")

    info = get_video_info(video)
    results.append(info)


df = pd.DataFrame(results)

# ハッシュが同じ動画を重複グループとして分類
df["duplicate_group"] = df.groupby("hash").ngroup()

# 同じハッシュが複数存在するか
hash_counts = df["hash"].value_counts()

df["is_duplicate"] = df["hash"].map(hash_counts) > 1

# 保存
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# 結果表示
duplicate_df = df[df["is_duplicate"]]

print()
print("=" * 60)
print("重複チェック完了")
print("=" * 60)

print(f"総動画数       : {len(df)}")
print(f"重複している動画: {len(duplicate_df)}")
print(f"重複を除いた動画: {df['hash'].nunique()}")

print()
print("重複グループ:")
print()

for group_id, group in duplicate_df.groupby("duplicate_group"):
    print(f"--- グループ {group_id} ---")

    for video in group["video"]:
        print(video)

    print()

print("結果保存先:")
print(OUTPUT_FILE)
