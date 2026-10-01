import cv2
from pathlib import Path

IMAGE_DIR = Path("ball_dataset/additional_frames/mlb_082")

LABEL_DIR = Path("ball_dataset/additional_labels/mlb_082")

LABEL_DIR.mkdir(parents=True, exist_ok=True)

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    print("画像が見つかりません")
    exit()

index = 0
drawing = False
start_x = 0
start_y = 0
end_x = 0
end_y = 0
current_box = None


def mouse_callback(event, x, y, flags, param):
    global drawing
    global start_x, start_y
    global end_x, end_y
    global current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        end_x = x
        end_y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x = x
            end_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x = x
        end_y = y

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        if x2 > x1 and y2 > y1:
            current_box = (x1, y1, x2, y2)


window_name = "Ball Labeling"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:

    if index >= len(images):
        break

    image_path = images[index]
    image = cv2.imread(str(image_path))

    if image is None:
        print("画像を読み込めません:", image_path)
        index += 1

        if index >= len(images):
            break

        continue

    current_box = None

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    while True:

        display = image.copy()

        # 既存ラベルがあれば表示
        if label_path.exists():

            lines = label_path.read_text(encoding="utf-8").strip().splitlines()

            for line in lines:

                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                _, cx, cy, w, h = map(float, parts)

                height, width = image.shape[:2]

                x1 = int((cx - w / 2) * width)
                y1 = int((cy - h / 2) * height)
                x2 = int((cx + w / 2) * width)
                y2 = int((cy + h / 2) * height)

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 現在描いている枠
        if drawing or current_box is not None:

            if current_box is not None:
                x1, y1, x2, y2 = current_box
            else:
                x1 = min(start_x, end_x)
                y1 = min(start_y, end_y)
                x2 = max(start_x, end_x)
                y2 = max(start_y, end_y)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 説明
        cv2.putText(
            display,
            f"{index + 1}/{len(images)}  " f"{image_path.name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "Drag: ball  S: save  N: skip  B: back  Q: quit",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # S = 保存
        if key == ord("s"):

            if current_box is not None:

                x1, y1, x2, y2 = current_box

                height, width = image.shape[:2]

                # YOLO形式に変換
                cx = ((x1 + x2) / 2) / width
                cy = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                label_path.write_text(
                    f"0 {cx:.6f} {cy:.6f} " f"{w:.6f} {h:.6f}\n", encoding="utf-8"
                )

                print(f"保存: {image_path.name}")

                index += 1
                break

            else:
                print("ボールを囲んでから S を押してください")

        # N = スキップ
        elif key == ord("n"):

            # 空のラベルを作る
            label_path.write_text("", encoding="utf-8")

            print(f"スキップ: {image_path.name}")

            index += 1
            break

        # B = 1つ戻る
        elif key == ord("b"):

            if index > 0:
                index -= 1

            break

        # Q = 終了
        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("ラベリングを終了しました")
            print("ラベル保存先:", LABEL_DIR.resolve())

            exit()

cv2.destroyAllWindows()

print()
print("=" * 50)
print("ラベリング完了")
print("画像数:", len(images))
print("ラベル保存先:", LABEL_DIR.resolve())
print("=" * 50)
