import cv2
from pathlib import Path

# ========================================
# 入力画像とラベル保存先
# ========================================

IMAGE_DIR = Path("output/mlb_007_missing_frames")
LABEL_DIR = Path("ball_dataset/missing_frame_labels")

LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# 画像一覧
# ========================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    print("画像がありません")
    raise SystemExit


# ========================================
# マウス操作用の変数
# ========================================

index = 0

drawing = False
start_x = 0
start_y = 0
box = None


# ========================================
# マウス操作
# ========================================


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global box

    # マウスを押した
    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        box = (x, y, x, y)

    # ドラッグ中
    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            box = (start_x, start_y, x, y)

    # マウスを離した
    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            box = (start_x, start_y, x, y)


# ========================================
# ウィンドウ
# ========================================

window_name = "Ball Label"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

cv2.setMouseCallback(window_name, mouse_callback)


# ========================================
# ラベル付け開始
# ========================================

while index < len(images):

    image_path = images[index]

    image = cv2.imread(str(image_path))

    if image is None:

        index += 1

        continue

    height, width = image.shape[:2]

    # その画像のラベルをリセット
    box = None
    drawing = False

    print()
    print(f"{index + 1}/{len(images)}: {image_path.name}")

    # ====================================
    # 画像を表示
    # ====================================

    while True:

        display = image.copy()

        # --------------------------------
        # ボックスを表示
        # --------------------------------

        if box is not None:

            x1, y1, x2, y2 = box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --------------------------------
        # 進捗表示
        # --------------------------------

        cv2.putText(
            display,
            f"{index + 1}/{len(images)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # --------------------------------
        # 操作説明
        # --------------------------------

        cv2.putText(
            display,
            "Drag ball | S save | N skip | B back | Q quit",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ====================================
        # S = 保存して次へ
        # ====================================

        if key == ord("s"):

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            # --------------------------------
            # ボールなし
            # --------------------------------

            if box is None:

                label_path.write_text("")

                print("→ ボールなし")

            # --------------------------------
            # ボールあり
            # --------------------------------

            else:

                x1, y1, x2, y2 = box

                left = min(x1, x2)
                right = max(x1, x2)

                top = min(y1, y2)
                bottom = max(y1, y2)

                box_width = right - left
                box_height = bottom - top

                # --------------------------------
                # 小さすぎる場合
                # --------------------------------

                if box_width < 2 or box_height < 2:

                    label_path.write_text("")

                    print("→ ボックスが小さすぎます")

                else:

                    # --------------------------------
                    # 中心座標
                    # --------------------------------

                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2

                    # --------------------------------
                    # YOLO形式に正規化
                    # --------------------------------

                    center_x /= width
                    center_y /= height

                    box_width /= width
                    box_height /= height

                    # --------------------------------
                    # YOLOラベル保存
                    # class 0 = ball
                    # --------------------------------

                    label_path.write_text(
                        f"0 "
                        f"{center_x:.6f} "
                        f"{center_y:.6f} "
                        f"{box_width:.6f} "
                        f"{box_height:.6f}\n"
                    )

                    print("→ ボール保存")

            index += 1

            break

        # ====================================
        # N = 保存せず次へ
        # ====================================

        elif key == ord("n"):

            index += 1

            break

        # ====================================
        # B = 前へ
        # ====================================

        elif key == ord("b"):

            if index > 0:
                index -= 1

            break

        # ====================================
        # Q = 終了
        # ====================================

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("終了しました")

            raise SystemExit


# ========================================
# 終了
# ========================================

cv2.destroyAllWindows()

print()
print("ラベル付け終了")
print("保存場所:", LABEL_DIR)
