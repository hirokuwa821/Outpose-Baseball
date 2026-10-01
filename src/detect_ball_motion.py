import cv2
import numpy as np
from pathlib import Path
import sys
import csv

# ============================================================
# 設定
# ============================================================

if len(sys.argv) < 2:
    print("使い方:")
    print("python src/detect_ball_motion.py videos\\pitch_102_1.mp4")
    sys.exit()

VIDEO_PATH = sys.argv[1]

OUTPUT_VIDEO = "runs/detect/ball_motion_debug.mp4"
OUTPUT_CSV = "output/ball_motion_candidates.csv"

# 投球動作が発生する範囲
# 今回の動画ではFrame 45～70付近を重点的に見る
START_FRAME = 35
END_FRAME = 80

# 白い物体の条件
MIN_AREA = 2
MAX_AREA = 500

# 白色判定
WHITE_VALUE = 150
WHITE_SATURATION = 100

# 前フレームとの差
DIFF_THRESHOLD = 25


# ============================================================
# 出力先
# ============================================================

Path("runs/detect").mkdir(parents=True, exist_ok=True)

Path("output").mkdir(parents=True, exist_ok=True)


# ============================================================
# 動画
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print()
print("==========================================")
print("ボール候補検出")
print("==========================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)
print("解析範囲:", START_FRAME, "～", END_FRAME)


# ============================================================
# 出力動画
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# ============================================================
# CSV
# ============================================================

candidates = []


# ============================================================
# 前フレーム
# ============================================================

previous_gray = None

frame_number = 0


# ============================================================
# フレーム処理
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    debug = frame.copy()

    # --------------------------------------------------------
    # 前後のフレームも動画として保存
    # --------------------------------------------------------

    if frame_number < START_FRAME or frame_number > END_FRAME:

        cv2.putText(
            debug,
            f"Frame {frame_number}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        out.write(debug)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        previous_gray = gray

        continue

    # ========================================================
    # グレースケール
    # ========================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # 動き検出
    # ========================================================

    if previous_gray is None:

        previous_gray = gray

        out.write(debug)

        continue

    diff = cv2.absdiff(gray, previous_gray)

    _, motion = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)

    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)

    # ========================================================
    # 白色領域
    # ========================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    white_mask = cv2.inRange(
        hsv, np.array([0, 0, WHITE_VALUE]), np.array([180, WHITE_SATURATION, 255])
    )

    # ========================================================
    # 動き AND 白色
    # ========================================================

    candidate_mask = cv2.bitwise_and(motion, white_mask)

    # ========================================================
    # 輪郭
    # ========================================================

    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    frame_candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w / 2
        cy = y + h / 2

        # ----------------------------------------------------
        # 小さな候補
        # ----------------------------------------------------

        if w > 80:
            continue

        if h > 80:
            continue

        frame_candidates.append(
            {
                "frame": frame_number,
                "cx": cx,
                "cy": cy,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
            }
        )

    # ========================================================
    # 候補描画
    # ========================================================

    for c in frame_candidates:

        x = int(c["x"])
        y = int(c["y"])
        w = int(c["w"])
        h = int(c["h"])

        cx = int(c["cx"])
        cy = int(c["cy"])

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            debug,
            "BALL?",
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        candidates.append(c)

    # ========================================================
    # 情報表示
    # ========================================================

    cv2.putText(
        debug,
        f"Frame {frame_number}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        debug,
        f"Candidates: {len(frame_candidates)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.rectangle(debug, (0, 0), (width - 1, height - 1), (255, 0, 0), 3)

    out.write(debug)

    previous_gray = gray


# ============================================================
# 終了
# ============================================================

cap.release()
out.release()


# ============================================================
# CSV保存
# ============================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "frame",
            "time_sec",
            "center_x",
            "center_y",
            "x",
            "y",
            "width",
            "height",
            "area",
        ]
    )

    for c in candidates:

        writer.writerow(
            [
                c["frame"],
                c["frame"] / fps,
                c["cx"],
                c["cy"],
                c["x"],
                c["y"],
                c["w"],
                c["h"],
                c["area"],
            ]
        )


# ============================================================
# 結果
# ============================================================

print()
print("==========================================")
print("検出結果")
print("==========================================")

print("ボール候補総数:", len(candidates))


frames = sorted(set(c["frame"] for c in candidates))


print("候補が出たフレーム:", frames)


print()
print("保存先:")
print("デバッグ動画:", OUTPUT_VIDEO)

print("CSV:", OUTPUT_CSV)

print()
print("==========================================")
print("完了")
print("==========================================")
