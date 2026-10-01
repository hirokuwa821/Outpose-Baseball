import cv2
import json
from pathlib import Path

# ==========================================
# 設定
# ==========================================

VIDEO_PATH = "test_videos/mlb_002_81.9mph_MM9QJ9EPUPHF_right.mp4"
OUTPUT_FILE = "output/calibration_points.json"

START_FRAME = 30

# A/Dを押したとき何フレーム進むか
FRAME_STEP = 5


# ==========================================
# 動画を開く
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_number = START_FRAME

points = []


# ==========================================
# フレーム取得
# ==========================================


def get_frame(frame_number):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        return None

    return frame


frame = get_frame(frame_number)

if frame is None:
    print("フレームを読み込めません")
    cap.release()
    exit()


# ==========================================
# マウスクリック
# ==========================================


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append({"x": x, "y": y})

        print(f"点 {len(points)}: " f"x={x}, y={y}")


# ==========================================
# ウィンドウ
# ==========================================

window_name = "Calibration"

cv2.namedWindow(window_name)

cv2.setMouseCallback(window_name, mouse_callback)


print()
print("==========================================")
print("キャリブレーション")
print("==========================================")
print()
print("A : 5フレーム戻る")
print("D : 5フレーム進む")
print("W : 1フレーム戻る")
print("E : 1フレーム進む")
print()
print("C : クリックした点を全部消す")
print("S : 保存")
print("ESC : 終了")
print()
print(f"総フレーム数 : {total_frames}")
print(f"FPS : {fps}")
print()
print("※ 最初に動画画面をクリックしてから")
print("  キーボードを操作してください。")
print()


# ==========================================
# メインループ
# ==========================================

while True:

    display = frame.copy()

    # --------------------------------------
    # クリックした点を表示
    # --------------------------------------

    for i, point in enumerate(points):

        x = point["x"]
        y = point["y"]

        cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            display,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # --------------------------------------
    # フレーム番号
    # --------------------------------------

    cv2.putText(
        display,
        f"Frame: {frame_number}/{total_frames - 1}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # --------------------------------------
    # 操作説明
    # --------------------------------------

    cv2.putText(
        display,
        "A/D: +/-5   W/E: +/-1   C: Clear   S: Save",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.imshow(window_name, display)

    # waitKeyExを使用
    key = cv2.waitKeyEx(30)

    # --------------------------------------
    # ESC
    # --------------------------------------

    if key == 27:
        break

    # --------------------------------------
    # A : 5フレーム戻る
    # --------------------------------------

    elif key == ord("a") or key == ord("A"):

        frame_number -= FRAME_STEP

        if frame_number < 0:
            frame_number = 0

        new_frame = get_frame(frame_number)

        if new_frame is not None:
            frame = new_frame

        print(f"現在のフレーム: {frame_number}")

    # --------------------------------------
    # D : 5フレーム進む
    # --------------------------------------

    elif key == ord("d") or key == ord("D"):

        frame_number += FRAME_STEP

        if frame_number >= total_frames:
            frame_number = total_frames - 1

        new_frame = get_frame(frame_number)

        if new_frame is not None:
            frame = new_frame

        print(f"現在のフレーム: {frame_number}")

    # --------------------------------------
    # W : 1フレーム戻る
    # --------------------------------------

    elif key == ord("w") or key == ord("W"):

        frame_number -= 1

        if frame_number < 0:
            frame_number = 0

        new_frame = get_frame(frame_number)

        if new_frame is not None:
            frame = new_frame

        print(f"現在のフレーム: {frame_number}")

    # --------------------------------------
    # E : 1フレーム進む
    # --------------------------------------

    elif key == ord("e") or key == ord("E"):

        frame_number += 1

        if frame_number >= total_frames:
            frame_number = total_frames - 1

        new_frame = get_frame(frame_number)

        if new_frame is not None:
            frame = new_frame

        print(f"現在のフレーム: {frame_number}")

    # --------------------------------------
    # C : 点を全部消す
    # --------------------------------------

    elif key == ord("c") or key == ord("C"):

        points.clear()

        print("クリックした点を全部消しました")

    # --------------------------------------
    # S : 保存
    # --------------------------------------

    elif key == ord("s") or key == ord("S"):

        if len(points) < 4:

            print()
            print(f"最低4点必要です。" f"現在 {len(points)}点です。")

            continue

        Path("output").mkdir(parents=True, exist_ok=True)

        data = {
            "video": VIDEO_PATH,
            "frame": frame_number,
            "fps": fps,
            "points": points,
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

            json.dump(data, f, indent=4, ensure_ascii=False)

        print()
        print("==========================================")
        print("保存完了")
        print("==========================================")
        print()
        print(f"保存先: {OUTPUT_FILE}")
        print(f"使用フレーム: {frame_number}")
        print(f"取得した点: {len(points)}点")
        print()

        for i, point in enumerate(points):

            print(f"{i + 1}: " f"x={point['x']}, " f"y={point['y']}")

        break


# ==========================================
# 終了
# ==========================================

cap.release()
cv2.destroyAllWindows()
