from pathlib import Path
import shutil

# ==========================================
# 設定
# ==========================================

EXTRA_IMAGE_DIR = Path("output/mlb_007_frames_130_150")
EXTRA_LABEL_DIR = Path("ball_dataset/mlb007_extra_labels")

DATASET_DIR = Path("ball_dataset/final_ball_dataset_v6")

TRAIN_IMAGE_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABEL_DIR = DATASET_DIR / "labels" / "train"

# ==========================================
# フォルダ確認
# ==========================================

if not EXTRA_IMAGE_DIR.exists():
    print("追加画像フォルダがありません")
    print(EXTRA_IMAGE_DIR)
    raise SystemExit

if not EXTRA_LABEL_DIR.exists():
    print("追加ラベルフォルダがありません")
    print(EXTRA_LABEL_DIR)
    raise SystemExit

TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 135～148を追加
# ==========================================

added_images = 0
added_labels = 0

for frame_no in range(135, 149):

    image_path = EXTRA_IMAGE_DIR / f"mlb_007_frame_{frame_no:03d}.jpg"

    label_path = EXTRA_LABEL_DIR / f"mlb_007_frame_{frame_no:03d}.txt"

    if not image_path.exists():
        print(f"画像なし: Frame {frame_no}")
        continue

    if not label_path.exists():
        print(f"ラベルなし: Frame {frame_no}")
        continue

    # trainへコピー
    destination_image = TRAIN_IMAGE_DIR / image_path.name

    destination_label = TRAIN_LABEL_DIR / label_path.name

    shutil.copy2(image_path, destination_image)

    shutil.copy2(label_path, destination_label)

    added_images += 1
    added_labels += 1

    print(f"追加: Frame {frame_no}")

# ==========================================
# 結果
# ==========================================

print()
print("=" * 50)
print("追加完了")
print("=" * 50)
print("追加画像:", added_images)
print("追加ラベル:", added_labels)
print("追加先:", DATASET_DIR)
