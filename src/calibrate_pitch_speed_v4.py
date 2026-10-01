import cv2
import pandas as pd
from pathlib import Path

# ============================================================
# 設定
# ============================================================

VIDEO_DIR = Path("videos")

# v3で作成した速度CSV
SPEED_DIR = Path("output/pitch_speed_v3")

# v4の出力先
OUTPUT_DIR = Path("output/pitch_speed_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 自分の実際の身長
HEIGHT_CM = 175.0

VIDEO_NAMES = [
    "pitch_102_1.mp4",
    "pitch_98_1.mp4",
    "pitch_98_2.mp4",
]


# ============================================================
# 身長のピクセル数を測定
# ============================================================


def measure_player_height(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("動画を開けません:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 最初は動画中央付近
    frame_no = total_frames // 2

    points = []

    window_name = "Height Calibration"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 2:

                points.append((x, y))

                print(f"クリック {len(points)}: " f"({x}, {y})")

    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print("============================================================")
    print("身長ピクセル測定")
    print("============================================================")
    print()
    print("動画:", video_path.name)
    print()
    print("頭頂部と足裏をクリックしてください。")
    print()
    print("操作:")
    print("  A : 1フレーム戻る")
    print("  D : 1フレーム進む")
    print("  R : クリックをリセット")
    print("  S : 測定確定")
    print("  Q : 中止")
    print()

    while True:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        # --------------------------------------------
        # クリックした点
        # --------------------------------------------

        for i, (x, y) in enumerate(points):

            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)

            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # --------------------------------------------
        # 2点あれば身長pxを表示
        # --------------------------------------------

        if len(points) == 2:

            x1, y1 = points[0]
            x2, y2 = points[1]

            height_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.putText(
                display,
                f"Height = {height_px:.1f} px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            display,
            f"Frame: {frame_no}/{total_frames - 1}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # --------------------------------------------
        # 前のフレーム
        # --------------------------------------------

        if key == ord("a"):

            frame_no = max(0, frame_no - 1)

            points.clear()

        # --------------------------------------------
        # 次のフレーム
        # --------------------------------------------

        elif key == ord("d"):

            frame_no = min(total_frames - 1, frame_no + 1)

            points.clear()

        # --------------------------------------------
        # リセット
        # --------------------------------------------

        elif key == ord("r"):

            points.clear()

            print("クリックをリセットしました")

        # --------------------------------------------
        # 保存
        # --------------------------------------------

        elif key == ord("s"):

            if len(points) != 2:

                print("頭頂部と足裏の2点をクリックしてください")

                continue

            x1, y1 = points[0]
            x2, y2 = points[1]

            height_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if height_px < 50:

                print("身長pxが小さすぎます。" "測定し直してください。")

                continue

            cv2.destroyWindow(window_name)

            cap.release()

            return height_px

        # --------------------------------------------
        # 終了
        # --------------------------------------------

        elif key == ord("q"):

            cv2.destroyAllWindows()
            cap.release()

            return None

    cv2.destroyAllWindows()
    cap.release()

    return None


# ============================================================
# px/s → km/h
# ============================================================


def px_per_sec_to_kmh(speed_px_s, height_px):

    # 1 px が何cmか
    cm_per_px = HEIGHT_CM / height_px

    # px/s → cm/s
    speed_cm_s = speed_px_s * cm_per_px

    # cm/s → km/h
    speed_kmh = speed_cm_s * 3600 / 100000

    return speed_kmh


# ============================================================
# メイン処理
# ============================================================

print()
print("============================================================")
print("投球速度解析 v4")
print("身長175 cmによるスケール換算")
print("============================================================")
print()

results = []


for video_name in VIDEO_NAMES:

    video_path = VIDEO_DIR / video_name

    stem = video_path.stem

    csv_path = SPEED_DIR / f"{stem}_speed_v3.csv"

    print()
    print("############################################################")
    print("動画:", video_name)
    print("############################################################")

    # --------------------------------------------------------
    # CSV確認
    # --------------------------------------------------------

    if not csv_path.exists():

        print("速度CSVがありません:")

        print(csv_path)

        continue

    # --------------------------------------------------------
    # 身長px測定
    # --------------------------------------------------------

    height_px = measure_player_height(video_path)

    if height_px is None:

        print("身長測定をスキップしました")

        continue

    print()
    print(f"画像上の身長: " f"{height_px:.2f} px")

    print(f"実際の身長: " f"{HEIGHT_CM:.1f} cm")

    # --------------------------------------------------------
    # CSV読み込み
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    print()
    print("CSV列:", list(df.columns))

    # --------------------------------------------------------
    # px/s列を探す
    # --------------------------------------------------------

    speed_column = None

    for column in ["speed_px_s", "speed", "velocity_px_s"]:

        if column in df.columns:

            speed_column = column

            break

    if speed_column is None:

        print()
        print("speed_px_s列が見つかりません")

        continue

    # --------------------------------------------------------
    # km/hへ変換
    # --------------------------------------------------------

    df["speed_kmh"] = df[speed_column].apply(
        lambda x: px_per_sec_to_kmh(float(x), height_px)
    )

    # --------------------------------------------------------
    # 平均・最大
    # --------------------------------------------------------

    average_speed = df["speed_kmh"].mean()

    max_speed = df["speed_kmh"].max()

    max_index = df["speed_kmh"].idxmax()

    max_row = df.loc[max_index]

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print(f"身長px        : " f"{height_px:.2f} px")

    print(f"平均速度      : " f"{average_speed:.2f} km/h")

    print(f"最大瞬間速度  : " f"{max_speed:.2f} km/h")

    if "frame1" in df.columns:

        print(
            f"最大速度区間  : "
            f"Frame "
            f"{int(max_row['frame1'])}"
            f" → "
            f"{int(max_row['frame2'])}"
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    output_csv = OUTPUT_DIR / f"{stem}_speed_v4.csv"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print()
    print("保存:", output_csv)

    results.append(
        {
            "video": stem,
            "height_px": height_px,
            "average": average_speed,
            "maximum": max_speed,
        }
    )


# ============================================================
# 最終結果
# ============================================================

print()
print("============================================================")
print("全動画の速度解析完了")
print("============================================================")
print()

for result in results:

    print(
        f"{result['video']} | "
        f"身長: {result['height_px']:.1f} px | "
        f"平均: {result['average']:.2f} km/h | "
        f"最大: {result['maximum']:.2f} km/h"
    )

print()
print("出力先:", OUTPUT_DIR)
