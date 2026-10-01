import cv2
from pathlib import Path

# ============================================================
# 設定
# ============================================================

IMAGE_DIR = Path("dataset/ball_annotation/images")
LABEL_DIR = Path("dataset/ball_annotation/labels")

OUTPUT_DIR = Path("dataset/ball_annotation/label_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 画像取得
# ============================================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    print("画像がありません")
    exit()

print("==========================================")
print("ボールラベル確認")
print("==========================================")
print("画像数:", len(images))
print("画像:", IMAGE_DIR)
print("ラベル:", LABEL_DIR)
print("出力:", OUTPUT_DIR)
print()

# ============================================================
# 各画像を処理
# ============================================================

labeled_count = 0
empty_count = 0

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print("画像を読み込めません:", image_path.name)
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # ラベル読み込み
    # --------------------------------------------------------

    boxes = []

    if label_path.exists():

        text = label_path.read_text(encoding="utf-8").strip()

        if text:

            for line in text.splitlines():

                values = line.split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                center_x = float(values[1])
                center_y = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                # YOLO座標 → pixel座標
                cx = center_x * width
                cy = center_y * height

                w = box_width * width
                h = box_height * height

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                # 画像範囲内に収める
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))

                boxes.append(
                    {
                        "class_id": class_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

    # --------------------------------------------------------
    # 枠を描画
    # --------------------------------------------------------

    display = image.copy()

    if boxes:

        labeled_count += 1

        for box in boxes:

            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            # ボールのラベル
            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                display,
                "BALL",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    else:

        empty_count += 1

        cv2.putText(
            display,
            "NO BALL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    # --------------------------------------------------------
    # ファイル保存
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        display,
    )

    print(f"{image_path.name:35s} " f"labels={len(boxes)}")

# ============================================================
# 結果
# ============================================================

print()
print("==========================================")
print("確認画像作成完了")
print("==========================================")

print("総画像数 :", len(images))
print("ボールあり:", labeled_count)
print("ボールなし:", empty_count)

print()
print("保存先:")
print(OUTPUT_DIR)
