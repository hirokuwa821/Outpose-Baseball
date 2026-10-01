from pathlib import Path
import shutil
import random
import pandas as pd

# ==========================================
# 設定
# ==========================================

OLD_DATASET = Path("ball_dataset/final_ball_dataset_v3")

MANUAL_ROOT = Path("ball_dataset/manual_labels")

VIDEO_LIST = Path("output/ball_label_candidates_20.csv")

OUTPUT_ROOT = Path("ball_dataset/final_ball_dataset_v4")

TRAIN_RATIO = 0.8

RANDOM_SEED = 42


# ==========================================
# 初期化
# ==========================================

random.seed(RANDOM_SEED)

for folder in [
    OUTPUT_ROOT / "images" / "train",
    OUTPUT_ROOT / "images" / "val",
    OUTPUT_ROOT / "labels" / "train",
    OUTPUT_ROOT / "labels" / "val",
]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 既存v3データをコピー
# ==========================================

print("=" * 60)
print("v3データをコピー")
print("=" * 60)

for split in ["train", "val"]:

    image_dir = OLD_DATASET / "images" / split

    label_dir = OLD_DATASET / "labels" / split

    output_image_dir = OUTPUT_ROOT / "images" / split

    output_label_dir = OUTPUT_ROOT / "labels" / split

    image_files = list(image_dir.glob("*.jpg"))

    for image_path in image_files:

        shutil.copy2(image_path, output_image_dir / image_path.name)

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():

            shutil.copy2(label_path, output_label_dir / label_path.name)

    print(split, ":", len(image_files), "枚コピー")


# ==========================================
# 今回の20動画を取得
# ==========================================

video_df = pd.read_csv(VIDEO_LIST)

videos = video_df["video"].tolist()

print()
print("=" * 60)
print("今回の20動画")
print("=" * 60)

for video in videos:
    print(video)


# ==========================================
# 20動画をtrain / valに分ける
# ==========================================

videos = videos.copy()

random.shuffle(videos)

train_count = int(len(videos) * TRAIN_RATIO)

train_videos = set(videos[:train_count])

val_videos = set(videos[train_count:])

print()
print("train動画数:", len(train_videos))
print("val動画数  :", len(val_videos))


# ==========================================
# 手動ラベルをコピー
# ==========================================

print()
print("=" * 60)
print("手動ラベルをコピー")
print("=" * 60)

train_images = 0
val_images = 0

train_positive = 0
val_positive = 0


for video in videos:

    video_stem = Path(video).stem

    source_dir = MANUAL_ROOT / video_stem

    if not source_dir.exists():

        print("ラベルフォルダなし:", video_stem)

        continue

    # train / val
    if video in train_videos:

        split = "train"

    else:

        split = "val"

    output_image_dir = OUTPUT_ROOT / "images" / split

    output_label_dir = OUTPUT_ROOT / "labels" / split

    # auto_candidatesから画像を探す
    candidate_dir = Path("ball_dataset/auto_candidates") / video_stem

    image_files = sorted(candidate_dir.glob("*.jpg"))

    for image_path in image_files:

        # ----------------------------------
        # 画像名に動画名を付ける
        # ----------------------------------

        new_stem = video_stem + "_" + image_path.stem

        output_image_path = output_image_dir / f"{new_stem}.jpg"

        output_label_path = output_label_dir / f"{new_stem}.txt"

        # ----------------------------------
        # 画像コピー
        # ----------------------------------

        shutil.copy2(image_path, output_image_path)

        # ----------------------------------
        # 手動ラベルコピー
        # ----------------------------------

        manual_label_path = MANUAL_ROOT / video_stem / f"{image_path.stem}.txt"

        if manual_label_path.exists():

            shutil.copy2(manual_label_path, output_label_path)

            text = manual_label_path.read_text(encoding="utf-8").strip()

            if text:
                if split == "train":
                    train_positive += 1
                else:
                    val_positive += 1

        else:

            # ラベルがない場合は空ラベル
            output_label_path.write_text("", encoding="utf-8")

        if split == "train":
            train_images += 1
        else:
            val_images += 1


# ==========================================
# data.yaml作成
# ==========================================

yaml_text = f"""path: {OUTPUT_ROOT.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: ball
"""

yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_path.write_text(yaml_text, encoding="utf-8")


# ==========================================
# 結果表示
# ==========================================

print()
print("=" * 60)
print("v4データセット作成完了")
print("=" * 60)

print()
print("既存v3 + 今回20動画")

print()
print("今回追加した画像")
print("train:", train_images)
print("val  :", val_images)

print()
print("今回追加したボールありラベル")
print("train:", train_positive)
print("val  :", val_positive)

print()
print("保存先:")
print(OUTPUT_ROOT)

print()
print("data.yaml:")
print(yaml_path)
