from pathlib import Path
import shutil
import random

ROOT = Path("ball_dataset")
OUTPUT = ROOT / "final_ball_dataset_v2"

# --------------------------------------------------
# 元データ
# --------------------------------------------------

# 81.1mph
PITCH1_IMAGES = ROOT / "yolo" / "images"
PITCH1_LABELS = ROOT / "yolo" / "labels"

# 81.5mph
PITCH2_IMAGES = ROOT / "pitch2" / "images"
PITCH2_LABELS = ROOT / "pitch2" / "labels"

# 82.5mph
PITCH3_IMAGES = ROOT / "new_pitch_images_3"
PITCH3_LABELS = ROOT / "new_pitch_labels_3"


# --------------------------------------------------
# 出力先
# --------------------------------------------------

TRAIN_IMAGES = OUTPUT / "images" / "train"
TRAIN_LABELS = OUTPUT / "labels" / "train"

VAL_IMAGES = OUTPUT / "images" / "val"
VAL_LABELS = OUTPUT / "labels" / "val"


# --------------------------------------------------
# フォルダ作成
# --------------------------------------------------

for folder in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 画像とラベルをコピーする関数
# --------------------------------------------------


def copy_dataset(images_dir, labels_dir, output_images, output_labels, prefix):

    images = list(images_dir.rglob("*.jpg"))

    count = 0
    positive = 0

    for image_path in images:

        # ラベルを探す
        label_candidates = list(labels_dir.rglob(f"{image_path.stem}.txt"))

        if not label_candidates:
            print("ラベルなし:", image_path.name)
            continue

        label_path = label_candidates[0]

        # ファイル名が重複しないようにする
        new_name = f"{prefix}_{image_path.name}"
        new_label_name = f"{prefix}_{image_path.stem}.txt"

        shutil.copy2(image_path, output_images / new_name)

        shutil.copy2(label_path, output_labels / new_label_name)

        count += 1

        # ラベルが空でなければボールあり
        if label_path.read_text().strip():
            positive += 1

    print(f"{prefix}: {count}枚 " f"(ボールあり {positive}枚)")

    return count, positive


# --------------------------------------------------
# データセット作成
# --------------------------------------------------

print("=" * 50)
print("ボール検出データセット v2 作成")
print("=" * 50)

print("\n【学習データ】")

train_total = 0
train_positive = 0

# 81.1mph → 学習
a, b = copy_dataset(PITCH1_IMAGES, PITCH1_LABELS, TRAIN_IMAGES, TRAIN_LABELS, "pitch1")

train_total += a
train_positive += b


# 82.5mph → 学習
a, b = copy_dataset(PITCH3_IMAGES, PITCH3_LABELS, TRAIN_IMAGES, TRAIN_LABELS, "pitch3")

train_total += a
train_positive += b


print("\n【検証データ】")

val_total = 0
val_positive = 0

# 81.5mph → 検証
a, b = copy_dataset(PITCH2_IMAGES, PITCH2_LABELS, VAL_IMAGES, VAL_LABELS, "pitch2")

val_total += a
val_positive += b


# --------------------------------------------------
# data.yaml
# --------------------------------------------------

yaml_path = OUTPUT / "data.yaml"

yaml_text = f"""path: {OUTPUT.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: ball
"""

yaml_path.write_text(yaml_text, encoding="utf-8")


# --------------------------------------------------
# 結果表示
# --------------------------------------------------

print("\n" + "=" * 50)
print("統合完了")
print("=" * 50)

print(f"学習画像: {train_total}")
print(f"学習 ボールあり: {train_positive}")

print(f"検証画像: {val_total}")
print(f"検証 ボールあり: {val_positive}")

print("\ndata.yaml:")
print(yaml_path)
