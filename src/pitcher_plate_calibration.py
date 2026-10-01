import cv2
import numpy as np
from pathlib import Path

# ============================================================
# ピッチャーズプレート キャリブレーション
# ============================================================

VIDEO_PATH = Path("videos/pitch_98_1.mp4")

OUTPUT_DIR = Path("output/plate_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "Pitcher Plate Calibration"


# ------------------------------------------------------------
# ピッチャーズプレートの実寸
#
# MLB/野球規格として使用
# 幅 24 inch = 60.96 cm
# 奥行き 6 inch = 15.24 cm
# ------------------------------------------------------------

PLATE_WIDTH_M = 0.6096
PLATE_DEPTH_M = 0.1524


# ============================================================
# 動画を開く
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("動画を開けません:")
    print(VIDEO_PATH)
    raise SystemExit


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = cap.get(cv2.CAP_PROP_FPS)


print()
print("============================================================")
print("ピッチャーズプレート キャリブレーション")
print("============================================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("総フレーム数:", total_frames)


# ============================================================
# フレームを取得
# ============================================================

# まず中央付近のフレームを表示
frame_number = total_frames // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

cap.release()


if not ret:
    print("フレーム取得失敗")
    raise SystemExit


height, width = frame.shape[:2]

print("サイズ:", width, "x", height)


# ============================================================
# マウス操作
# ============================================================

points = []


def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append((x, y))

            print(f"Point {len(points)}: " f"({x}, {y})")


# ============================================================
# ウィンドウ
# ============================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.resizeWindow(WINDOW_NAME, 960, 540)

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)


# ============================================================
# 4点指定
# ============================================================

print()
print("============================================================")
print("ピッチャーズプレートの4隅をクリックしてください")
print("============================================================")
print()
print("順番:")
print("1. 左上")
print("2. 右上")
print("3. 右下")
print("4. 左下")
print()
print("R : やり直し")
print("S : 保存")
print("Q : 終了")
print()


while True:

    display = frame.copy()

    # --------------------------------------------------------
    # 点を描画
    # --------------------------------------------------------

    for i, (x, y) in enumerate(points):

        cv2.circle(display, (x, y), 7, (0, 0, 255), -1)

        cv2.putText(
            display,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # 線を描画
    # --------------------------------------------------------

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(display, points[i], points[i + 1], (255, 0, 0), 2)

    if len(points) == 4:

        cv2.line(display, points[3], points[0], (255, 0, 0), 2)

    # --------------------------------------------------------
    # 説明
    # --------------------------------------------------------

    cv2.putText(
        display,
        "Click plate corners: TL -> TR -> BR -> BL",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        "R: reset   S: save   Q: quit",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------------------------------
    # R
    # --------------------------------------------------------

    if key == ord("r"):

        points = []

        print("リセット")

    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    elif key == ord("q"):

        break

    # --------------------------------------------------------
    # S
    # --------------------------------------------------------

    elif key == ord("s"):

        if len(points) != 4:

            print("4点すべて指定してください")

            continue

        # ====================================================
        # 実世界座標
        #
        # 左上 → 右上 → 右下 → 左下
        # ====================================================

        real_points = np.array(
            [
                [0.0, 0.0],
                [PLATE_WIDTH_M, 0.0],
                [PLATE_WIDTH_M, PLATE_DEPTH_M],
                [0.0, PLATE_DEPTH_M],
            ],
            dtype=np.float32,
        )

        image_points = np.array(points, dtype=np.float32)

        # ====================================================
        # ホモグラフィ行列
        # ====================================================

        H, mask = cv2.findHomography(image_points, real_points)

        if H is None:

            print("ホモグラフィ行列の計算に失敗")

            continue

        # ====================================================
        # 保存
        # ====================================================

        output_file = OUTPUT_DIR / "pitcher_plate_calibration.npz"

        np.savez(
            output_file,
            H=H,
            image_points=image_points,
            real_points=real_points,
            frame_number=frame_number,
            fps=fps,
        )

        # ====================================================
        # 確認用画像
        # ====================================================

        warped = cv2.warpPerspective(frame, H, (800, 400))

        warped_path = OUTPUT_DIR / "plate_warped.jpg"

        cv2.imwrite(str(warped_path), warped)

        print()
        print("============================================================")
        print("キャリブレーション保存完了")
        print("============================================================")

        print("設定:", output_file)

        print("確認画像:", warped_path)

        print()
        print("プレートの4点:")
        print(image_points)

        break


cv2.destroyAllWindows()
