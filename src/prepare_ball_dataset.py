from pathlib import Path
import shutil
import random

# 元画像・ラベル
IMAGE_DIR = Path("ball_dataset/images")
LABEL_DIR = Path("ball_dataset/labels")

# YOLO用データセット
BASE_DIR = Path("ball_dataset/yolo")

TRAIN_IMAGES = BASE_DIR / "images/train"
VAL_IMAGES = BASE_DIR / "images/val"

TRAIN_LABELS = BASE_DIR / "labels/train"
VAL_LABELS = BASE_DIR / "labels/val"

for directory in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
    directory.mkdir(parents=True, exist_ok=True)


# 画像一覧
images = sorted(IMAGE_DIR.glob("*.jpg"))

print("画像数:", len(images))

if len(images) == 0:
    print("画像がありません")
    exit()


# シャッフル
random.seed(42)
random.shuffle(images)

# 80%
split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

print("学習:", len(train_images))
print("検証:", len(val_images))


def copy_dataset(images, image_dest, label_dest):

    for image_path in images:

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        # 画像
        shutil.copy2(image_path, image_dest / image_path.name)

        # ラベル
        if label_path.exists():

            shutil.copy2(label_path, label_dest / label_path.name)

        else:

            # ラベルがなければ空ファイル
            (label_dest / f"{image_path.stem}.txt").write_text("")


copy_dataset(train_images, TRAIN_IMAGES, TRAIN_LABELS)

copy_dataset(val_images, VAL_IMAGES, VAL_LABELS)


print()
print("データセット作成完了")
print("場所:", BASE_DIR)
