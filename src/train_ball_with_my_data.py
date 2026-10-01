from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil

# ============================================================
# 設定
# ============================================================

BASE_MODEL = "runs/detect/train-2/weights/best.pt"

SOURCE_IMAGES = Path("dataset/ball_annotation/images")
SOURCE_LABELS = Path("dataset/ball_annotation/labels")

DATASET_DIR = Path("dataset/ball_finetune")

TRAIN_IMAGES = DATASET_DIR / "images/train"
TRAIN_LABELS = DATASET_DIR / "labels/train"

DATA_YAML = DATASET_DIR / "data.yaml"

# ============================================================
# フォルダ作成
# ============================================================

TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 画像・ラベルをコピー
# ============================================================

images = sorted(SOURCE_IMAGES.glob("*.jpg"))

if not images:
    print("画像がありません")
    exit()

print("==========================================")
print("自分の投球データセット作成")
print("==========================================")

copied = 0
labeled = 0
empty = 0

for image_path in images:

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        print("ラベルなし:", image_path.name)
        continue

    shutil.copy2(image_path, TRAIN_IMAGES / image_path.name)

    shutil.copy2(label_path, TRAIN_LABELS / label_path.name)

    copied += 1

    if label_path.stat().st_size > 0:
        labeled += 1
    else:
        empty += 1

print()
print("コピー画像:", copied)
print("ボールあり:", labeled)
print("ボールなし:", empty)

# ============================================================
# data.yaml
# ============================================================

data_config = {
    "path": str(DATASET_DIR.resolve()),
    "train": "images/train",
    "val": "images/train",
    "nc": 1,
    "names": ["ball"],
}

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_config, f, allow_unicode=True, sort_keys=False)

print()
print("data.yaml:")
print(DATA_YAML)

# ============================================================
# モデル読み込み
# ============================================================

print()
print("==========================================")
print("既存モデル読み込み")
print("==========================================")

model = YOLO(BASE_MODEL)

print("モデル:", BASE_MODEL)

# ============================================================
# 追加学習
# ============================================================

print()
print("==========================================")
print("追加学習開始")
print("==========================================")

results = model.train(
    data=str(DATA_YAML),
    # 既存モデルをベースに短時間だけ追加学習
    epochs=50,
    imgsz=640,
    batch=4,
    patience=15,
    # 小規模データなので学習率を低めにする
    lr0=0.0005,
    # 自分の画像を学習するので過度な変形を抑える
    degrees=0.0,
    translate=0.05,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    # 左右反転は投球方向を変えるため使用しない
    fliplr=0.0,
    # 上下反転もしない
    flipud=0.0,
    mosaic=0.0,
    project="runs/detect",
    name="ball_finetune_mydata",
    exist_ok=True,
    verbose=True,
)

# ============================================================
# 完了
# ============================================================

print()
print("==========================================")
print("追加学習完了")
print("==========================================")

print()
print("学習結果:")
print("runs/detect/ball_finetune_mydata")

print()
print("モデル:")
print("runs/detect/ball_finetune_mydata/weights/best.pt")
