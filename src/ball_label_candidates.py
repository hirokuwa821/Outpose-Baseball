from pathlib import Path
import cv2
import pandas as pd

# ==========================================
# 設定
# ==========================================

ROOT_DIR = Path("ball_dataset/auto_candidates")
OUTPUT_ROOT = Path("ball_dataset/manual_labels")

VIDEO_LIST = Path("output/ball_label_candidates_20.csv")


# ==========================================
# フォルダ一覧
# ==========================================

folders = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])


print("=" * 60)
print("ボール候補画像のラベル付け")
print("=" * 60)

print("対象動画数:", len(folders))

if len(folders) == 0:
    print("候補画像フォルダがありません。")
    exit()


# ==========================================
# マウス操作用変数
# ==========================================

drawing = False
start_x = 0
start_y = 0

current_box = None


def mouse_callback(event, x, y, flags, param):
    global drawing
    global start_x
    global start_y
    global current_box

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            current_box = (start_x, start_y, x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        current_box = (start_x, start_y, x, y)


# ==========================================
# 1動画ずつ処理
# ==========================================

for video_number, folder in enumerate(folders, start=1):

    print()
    print("=" * 60)
    print(f"{video_number}/{len(folders)} : {folder.name}")
    print("=" * 60)

    image_files = sorted(folder.glob("*.jpg"))

    if len(image_files) == 0:
        print("画像がありません。")
        continue

    # 保存先
    output_dir = OUTPUT_ROOT / folder.name

    output_dir.mkdir(parents=True, exist_ok=True)

    # ウィンドウ
    window_name = "Ball Label"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.setMouseCallback(window_name, mouse_callback)

    image_index = 0

    while image_index < len(image_files):

        image_path = image_files[image_index]

        image = cv2.imread(str(image_path))

        if image is None:
            image_index += 1
            continue

        display = image.copy()

        # 現在のボックスを表示
        if current_box is not None:

            x1, y1, x2, y2 = current_box

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 情報表示
        text = (
            f"{video_number}/{len(folders)}  " f"{image_index + 1}/{len(image_files)}"
        )

        cv2.putText(
            display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ----------------------------------
        # S = 保存
        # ----------------------------------

        if key == ord("s"):

            if current_box is None:

                # ボールなし
                label_path = output_dir / f"{image_path.stem}.txt"

                label_path.write_text("", encoding="utf-8")

                print(f"{image_path.name} : ボールなし")

            else:

                x1, y1, x2, y2 = current_box

                # 座標を正規化
                h, w = image.shape[:2]

                left = min(x1, x2)
                right = max(x1, x2)

                top = min(y1, y2)
                bottom = max(y1, y2)

                center_x = ((left + right) / 2) / w

                center_y = ((top + bottom) / 2) / h

                box_w = (right - left) / w

                box_h = (bottom - top) / h

                label_path = output_dir / f"{image_path.stem}.txt"

                label_text = (
                    f"0 "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{box_w:.6f} "
                    f"{box_h:.6f}\n"
                )

                label_path.write_text(label_text, encoding="utf-8")

                print(f"{image_path.name} : " "ボールあり")

            current_box = None
            image_index += 1

        # ----------------------------------
        # N = 次の画像
        # ----------------------------------

        elif key == ord("n"):

            current_box = None
            image_index += 1

        # ----------------------------------
        # B = 1枚戻る
        # ----------------------------------

        elif key == ord("b"):

            current_box = None

            if image_index > 0:
                image_index -= 1

        # ----------------------------------
        # Q = 終了
        # ----------------------------------

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("途中終了しました。")
            print("次回は最初からではなく、")
            print("未処理分を確認して続けます。")

            exit()

    cv2.destroyAllWindows()

    print(f"{folder.name} 完了")


print()
print("=" * 60)
print("20本すべてのラベル付け完了")
print("=" * 60)

print()
print("保存先:")
print(OUTPUT_ROOT)
