from ultralytics import YOLO
from pathlib import Path
import shutil

# ==========================================
# 設定
# ==========================================

BASE_MODEL = "runs/detect/runs/detect/ball_finetune_mydata/weights/best.pt"

IMAGE_DIR = Path("ball_dataset/additional_images")
LABEL_DIR = Path("ball_dataset/additional_labels")

DATASET_DIR = Path("ball_dataset/finetune_v2")

RUNS_DIR = "runs/detect/ball_finetune_mydata_v2"

EPOCHS = 80
IMG_SIZE = 1280
BATCH = 4


# ==========================================
# 確認
# ==========================================

print("=" * 60)
print("ボール検出モデル 追加学習 v2")
print("=" * 60)

print("ベースモデル:")
print(BASE_MODEL)

print()
print("画像:")
print(IMAGE_DIR)

print()
print("ラベル:")
print(LABEL_DIR)


if not Path(BASE_MODEL).exists():

    print()
    print("ERROR: ベースモデルがありません")
    print(BASE_MODEL)
    exit()


# ==========================================
# 画像・ラベル確認
# ==========================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:

    print()
    print("ERROR: 追加画像がありません")
    exit()


print()
print("画像数:", len(images))


valid_pairs = []
missing_labels = []

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():

        valid_pairs.append((image_path, label_path))

    else:

        missing_labels.append(image_path.name)


print("ラベルあり:", len(valid_pairs))
print("ラベルなし:", len(missing_labels))


if missing_labels:

    print()
    print("ラベルがない画像:")

    for name in missing_labels[:20]:
        print(" ", name)


if len(valid_pairs) == 0:

    print()
    print("ERROR: 使用できる画像・ラベルの組がありません")
    exit()


# ==========================================
# データセットフォルダ作成
# ==========================================

train_images = DATASET_DIR / "images" / "train"
train_labels = DATASET_DIR / "labels" / "train"

train_images.mkdir(parents=True, exist_ok=True)

train_labels.mkdir(parents=True, exist_ok=True)


# ==========================================
# データコピー
# ==========================================

print()
print("データセット作成中...")


for image_path, label_path in valid_pairs:

    shutil.copy2(image_path, train_images / image_path.name)

    shutil.copy2(label_path, train_labels / label_path.name)


print("データコピー完了")


# ==========================================
# data.yaml
# ==========================================

yaml_path = DATASET_DIR / "data.yaml"

yaml_text = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/train

names:
  0: ball
"""

yaml_path.write_text(yaml_text, encoding="utf-8")


print()
print("data.yaml:")
print(yaml_path)


# ==========================================
# モデル読み込み
# ==========================================

print()
print("=" * 60)
print("モデル読み込み")
print("=" * 60)

model = YOLO(BASE_MODEL)

print("読み込み完了")


# ==========================================
# 追加学習
# ==========================================

print()
print("=" * 60)
print("追加学習開始")
print("=" * 60)

print("Epochs:", EPOCHS)
print("Image size:", IMG_SIZE)
print("Batch:", BATCH)


results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project="runs/detect",
    name="ball_finetune_mydata_v2",
    exist_ok=True,
    # 小さいボール向け
    scale=0.3,
    translate=0.05,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    # 左右反転は投球動画では不要
    fliplr=0.0,
    # 明るさなどの変化
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,
    workers=0,
    patience=20,
    verbose=True,
)


# ==========================================
# 完了
# ==========================================

print()
print("=" * 60)
print("追加学習完了")
print("=" * 60)

print()
print("学習結果:")
print("runs/detect/ball_finetune_mydata_v2")

print()
print("モデル:")
print("runs/detect/ball_finetune_mydata_v2/weights/best.pt")
