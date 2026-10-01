from pathlib import Path
import shutil
import yaml
import pandas as pd

BASE_DATASET = Path("ball_dataset/final_ball_dataset_v4")
AUTO_CANDIDATES = Path("ball_dataset/auto_candidates")
MANUAL_LABELS = Path("ball_dataset/manual_labels")
CANDIDATES_FILE = Path("output/ball_label_candidates_20.csv")
V6_DATASET = Path("ball_dataset/final_ball_dataset_v6")

TRAIN_IMAGES = V6_DATASET / "images" / "train"
TRAIN_LABELS = V6_DATASET / "labels" / "train"


if V6_DATASET.exists():
    print("既存のv6を削除します...")
    shutil.rmtree(V6_DATASET)


print("v4をv6へコピーしています...")
shutil.copytree(BASE_DATASET, V6_DATASET)


candidates = pd.read_csv(CANDIDATES_FILE)

added_images = 0
added_labels = 0
negative_images = 0
video_count = 0


for _, row in candidates.iterrows():

    video_name = Path(row["video"]).stem

    image_dir = Path("ball_dataset/auto_candidates") / video_name
    label_dir = MANUAL_LABELS / video_name

    if not image_dir.exists():
        print(f"画像フォルダなし: {video_name}")
        continue

    if not label_dir.exists():
        print(f"ラベルフォルダなし: {video_name}")
        continue

    video_count += 1

    for image_path in sorted(image_dir.glob("*.jpg")):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"ラベルなし: {video_name}/{image_path.name}")
            continue

        new_name = f"{video_name}_{image_path.name}"

        destination_image = TRAIN_IMAGES / new_name
        destination_label = TRAIN_LABELS / f"{video_name}_{image_path.stem}.txt"

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

        added_images += 1

        label_text = label_path.read_text(encoding="utf-8").strip()

        if label_text == "":
            negative_images += 1
        else:
            added_labels += 1


print()
print("=" * 60)
print("v6データセット作成完了")
print("=" * 60)
print(f"追加動画数: {video_count}")
print(f"追加画像数: {added_images}")
print(f"追加ボールラベル: {added_labels}")
print(f"追加ネガティブ画像: {negative_images}")


data_yaml = {
    "path": str(V6_DATASET.resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "ball"},
}

with open(V6_DATASET / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)


print()
print("data.yaml:")
print((V6_DATASET / "data.yaml").resolve())
print("=" * 60)
