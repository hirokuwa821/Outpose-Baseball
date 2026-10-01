import cv2
from pathlib import Path

# ==========================================
# 設定
# ==========================================

IMAGE_DIR = Path("output/mlb_007_frames_130_150")

LABEL_DIR = Path("ball_dataset/mlb007_extra_labels")
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ボールが見えるフレーム
TARGET_FRAMES = range(135, 149)

# ==========================================
# マウス操作用変数
# ==========================================

drawing = False
start_x = 0
start_y = 0
box = None


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
# 対象画像を取得
# ==========================================

images = []

for frame_no in TARGET_FRAMES:

    image_path = IMAGE_DIR / f"mlb_007_frame_{frame_no:03d}.jpg"

    if image_path.exists():
        images.append((frame_no, image_path))
    else:
        print(f"画像なし: Frame {frame_no}")


if not images:

    print("対象画像がありません")
    raise SystemExit


# ==========================================
# ウィンドウ
# ==========================================

window_name = "MLB007 Ball Label"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

cv2.setMouseCallback(window_name, mouse_callback)


index = 0


# ==========================================
# ラベル付け
# ==========================================

while index < len(images):

    frame_no, image_path = images[index]

    image = cv2.imread(str(image_path))

    if image is None:

        print(f"読み込み失敗: Frame {frame_no}")

        index += 1
        continue

    height, width = image.shape[:2]

    # --------------------------------------
    # 既存ラベル確認
    # --------------------------------------

    label_path = LABEL_DIR / f"mlb_007_frame_{frame_no:03d}.txt"

    # 既存ラベルがあれば読み込む
    box = None

    if label_path.exists():

        text = label_path.read_text().strip()

        if text:

            values = text.split()

            if len(values) == 5:

                _, cx, cy, bw, bh = map(float, values)

                # YOLO座標 → pixel座標
                center_x = cx * width
                center_y = cy * height

                box_width = bw * width
                box_height = bh * height

                left = int(center_x - box_width / 2)

                right = int(center_x + box_width / 2)

                top = int(center_y - box_height / 2)

                bottom = int(center_y + box_height / 2)

                box = (left, top, right, bottom)

    print()
    print(f"{index + 1}/{len(images)} " f"Frame {frame_no}")

    if label_path.exists():
        print("既存ラベルがあります")
        print("S = 保存して次へ")
    else:
        print("新規ラベル")
        print("S = 保存して次へ")

    print(
        "操作: "
        "ドラッグ=ボール指定 / "
        "S=保存 / "
        "N=ボールなし / "
        "B=戻る / "
        "Q=終了"
    )

    # ======================================
    # 1フレームの操作
    # ======================================

    while True:

        display = image.copy()

        # ----------------------------------
        # ボックス表示
        # ----------------------------------

        if box is not None:

            x1, y1, x2, y2 = box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ----------------------------------
        # Frame番号
        # ----------------------------------

        cv2.putText(
            display,
            f"Frame {frame_no}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # ----------------------------------
        # 操作説明
        # ----------------------------------

        cv2.putText(
            display,
            "Drag ball | S save | N none | B back | Q quit",
            (10, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ==================================
        # S 保存
        # ==================================

        if key == ord("s"):

            if box is None:

                print("→ ボックスなし")

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

                else:

                    center_x = (left + right) / 2

                    center_y = (top + bottom) / 2

                    # YOLO形式に正規化

                    center_x /= width
                    center_y /= height

                    box_width /= width
                    box_height /= height

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

        # ==================================
        # N ボールなし
        # ==================================

        elif key == ord("n"):

            label_path.write_text("")

            print("→ ボールなしで保存")

            index += 1
            break

        # ==================================
        # B 前へ
        # ==================================

        elif key == ord("b"):

            if index > 0:
                index -= 1

            break

        # ==================================
        # Q 終了
        # ==================================

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("終了しました")
            print("保存場所:", LABEL_DIR)

            raise SystemExit


# ==========================================
# 終了
# ==========================================

cv2.destroyAllWindows()

print()
print("=" * 50)
print("ラベル付け終了")
print("=" * 50)
print("保存場所:", LABEL_DIR)
