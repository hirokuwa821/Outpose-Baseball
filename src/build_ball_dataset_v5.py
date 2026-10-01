from pathlib import Path
import shutil
import yaml

# =========================
# 元データ
# =========================
BASE_DATASET = Path("ball_dataset/final_ball_dataset_v4")

# 今回追加するデータ
ADDITIONAL_IMAGES = Path("ball_dataset/additional_frames/mlb_082")

ADDITIONAL_LABELS = Path("ball_dataset/additional_labels/mlb_082")

# =========================
# 新しいデータセット
# =========================
V5_DATASET = Path("ball_dataset/final_ball_dataset_v5")

TRAIN_IMAGES = V5_DATASET / "images" / "train"
TRAIN_LABELS = V5_DATASET / "labels" / "train"

VAL_IMAGES = V5_DATASET / "images" / "val"
VAL_LABELS = V5_DATASET / "labels" / "val"

# =========================
# v4をコピー
# =========================
if V5_DATASET.exists():
    print("既存のv5を削除します...")
    shutil.rmtree(V5_DATASET)

print("v4をv5へコピーしています...")

shutil.copytree(BASE_DATASET, V5_DATASET)

# =========================
# 追加データをtrainへ追加
# =========================
added_images = 0
added_labels = 0
negative_images = 0

for image_path in sorted(ADDITIONAL_IMAGES.glob("*.jpg")):

    # 画像
    destination_image = TRAIN_IMAGES / f"additional_{image_path.name}"

    shutil.copy2(image_path, destination_image)

    added_images += 1

    # 対応するラベル
    label_path = ADDITIONAL_LABELS / f"{image_path.stem}.txt"

    destination_label = TRAIN_LABELS / f"additional_{image_path.stem}.txt"

    if label_path.exists():

        shutil.copy2(label_path, destination_label)

        # 空ファイルならネガティブ画像
        if label_path.read_text(encoding="utf-8").strip() == "":
            negative_images += 1

        else:
            added_labels += 1

print()
print("=" * 50)
print("v5データセット作成完了")
print("=" * 50)
print("追加画像:", added_images)
print("追加ボールラベル:", added_labels)
print("追加ネガティブ画像:", negative_images)

# =========================
# data.yamlを作成
# =========================
data_yaml = {
    "path": str(V5_DATASET.resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "ball"},
}

yaml_path = V5_DATASET / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

print()
print("data.yaml:")
print(yaml_path.resolve())
print("=" * 50)
