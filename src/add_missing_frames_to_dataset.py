from pathlib import Path
import shutil

# ========================================
# 今回追加する画像
# ========================================

IMAGE_DIR = Path("output/mlb_007_missing_frames")
LABEL_DIR = Path("ball_dataset/missing_frame_labels")


# ========================================
# 追加先
# ========================================

DATASET_DIR = Path("ball_dataset/final_ball_dataset_v6")

TRAIN_IMAGES_DIR = DATASET_DIR / "images/train"
TRAIN_LABELS_DIR = DATASET_DIR / "labels/train"


TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# 画像を追加
# ========================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    print("追加する画像がありません")
    raise SystemExit


image_count = 0
label_count = 0


for image_path in images:

    # 画像をコピー
    destination_image = TRAIN_IMAGES_DIR / image_path.name

    shutil.copy2(image_path, destination_image)

    image_count += 1

    # 対応するラベル
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():

        destination_label = TRAIN_LABELS_DIR / label_path.name

        shutil.copy2(label_path, destination_label)

        # 中身が空でなければボールあり
        if label_path.read_text().strip():
            label_count += 1


# ========================================
# 結果
# ========================================

print()
print("追加完了")
print("追加画像:", image_count)
print("ボールありラベル:", label_count)
print("ボールなし画像:", image_count - label_count)
print("追加先:", DATASET_DIR)
