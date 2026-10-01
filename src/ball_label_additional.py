import cv2
from pathlib import Path

# ==========================================
# 自分の投球画像だけを使用
# ==========================================

IMAGE_DIR = Path("ball_dataset/additional_images")
LABEL_DIR = Path("ball_dataset/additional_labels")

LABEL_DIR.mkdir(parents=True, exist_ok=True)

# JPGだけ
images = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.JPG"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.JPEG"))
)

print("==========================================")
print("自分の投球画像 ラベリング")
print("==========================================")

print("画像フォルダ:")
print(IMAGE_DIR.resolve())

print()
print("ラベル保存先:")
print(LABEL_DIR.resolve())

print()
print("画像枚数:", len(images))

if not images:
    print()
    print("画像がありません")
    print("フォルダを確認してください:")
    print(IMAGE_DIR.resolve())
    exit()

# ==========================================
# 画像名確認
# ==========================================

print()
print("使用する画像:")

for i, image_path in enumerate(images):
    print(f"{i + 1:3d}: {image_path.name}")

print()
print("==========================================")

# ==========================================
# ラベリング変数
# ==========================================

index = 0

drawing = False
start_x = 0
start_y = 0
box = None


# ==========================================
# マウス操作
# ==========================================


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global box

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        box = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:
            box = (start_x, start_y, x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            box = (start_x, start_y, x, y)


# ==========================================
# ウィンドウ
# ==========================================

window_name = "MY BALL LABEL"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.setMouseCallback(window_name, mouse_callback)


# ==========================================
# ラベリング開始
# ==========================================

while index < len(images):

    image_path = images[index]

    image = cv2.imread(str(image_path))

    if image is None:

        print("画像読み込み失敗:", image_path)

        index += 1

        continue

    height, width = image.shape[:2]

    box = None
    drawing = False

    print()
    print("==========================================")
    print(f"{index + 1}/{len(images)}")
    print("画像:", image_path.name)
    print("==========================================")

    while True:

        display = image.copy()

        # ==================================
        # ボックス表示
        # ==================================

        if box is not None:

            x1, y1, x2, y2 = box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ==================================
        # 上部表示
        # ==================================

        cv2.putText(
            display,
            f"{index + 1}/{len(images)}  {image_path.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ==================================
        # 操作説明
        # ==================================

        cv2.putText(
            display,
            "Drag BALL | S=Save | N=No ball | B=Back | Q=Quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ==================================
        # S 保存
        # ==================================

        if key == ord("s"):

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            # ------------------------------
            # ボールなし
            # ------------------------------

            if box is None:

                label_path.write_text("", encoding="utf-8")

                print("→ ボールなし")

            # ------------------------------
            # ボールあり
            # ------------------------------

            else:

                x1, y1, x2, y2 = box

                left = min(x1, x2)
                right = max(x1, x2)

                top = min(y1, y2)
                bottom = max(y1, y2)

                box_width = right - left
                box_height = bottom - top

                if box_width < 2 or box_height < 2:

                    print("→ ボックスが小さすぎます")

                    continue

                center_x = (left + right) / 2

                center_y = (top + bottom) / 2

                # YOLO形式
                center_x /= width
                center_y /= height

                box_width /= width
                box_height /= height

                label_text = (
                    f"0 "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

                label_path.write_text(label_text, encoding="utf-8")

                print("→ ボール保存")
                print("→", label_text.strip())

            index += 1

            break

        # ==================================
        # N 次へ
        # ==================================

        elif key == ord("n"):

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            label_path.write_text("", encoding="utf-8")

            print("→ ボールなしとして保存")

            index += 1

            break

        # ==================================
        # B 前へ
        # ==================================

        elif key == ord("b"):

            if index > 0:

                index -= 1

                print("→ 前の画像へ")

            break

        # ==================================
        # Q 終了
        # ==================================

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("途中終了")

            print("ラベル保存先:", LABEL_DIR.resolve())

            exit()


# ==========================================
# 終了
# ==========================================

cv2.destroyAllWindows()

print()
print("==========================================")
print("ラベル付け終了")
print("==========================================")

print("画像フォルダ:")
print(IMAGE_DIR.resolve())

print()
print("ラベル保存先:")
print(LABEL_DIR.resolve())

print()
print("処理枚数:", len(images))
