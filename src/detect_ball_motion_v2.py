import cv2
import numpy as np
from pathlib import Path
import sys
import csv
import math

# ============================================================
# 設定
# ============================================================

if len(sys.argv) < 2:
    print("使い方:")
    print("python src/detect_ball_motion_v2.py videos\\pitch_102_1.mp4")
    sys.exit()

VIDEO_PATH = sys.argv[1]

OUTPUT_VIDEO = "runs/detect/ball_motion_debug_v2.mp4"
OUTPUT_CSV = "output/ball_motion_candidates_v2.csv"

# 今回の動画
START_FRAME = 35
END_FRAME = 80

# 白色判定
MIN_V = 160
MAX_S = 90

# ボールサイズ
MIN_AREA = 3
MAX_AREA = 250

# 大きすぎる候補を除外
MAX_WIDTH = 30
MAX_HEIGHT = 30

# 円形度
MIN_CIRCULARITY = 0.20

# 1フレームに表示する候補数
TOP_N = 10


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
print("ボール候補検出 v2")
print("==========================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)
print("解析範囲:", START_FRAME, "～", END_FRAME)


# ============================================================
# VideoWriter
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# ============================================================
# CSV
# ============================================================

all_candidates = []


# ============================================================
# 前フレーム
# ============================================================

previous_gray = None
previous_candidates = []

frame_number = 0


# ============================================================
# 距離
# ============================================================


def distance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ============================================================
# 円形度
# ============================================================


def circularity(area, perimeter):

    if perimeter <= 0:
        return 0

    return 4.0 * math.pi * area / (perimeter * perimeter)


# ============================================================
# フレーム処理
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    debug = frame.copy()

    # ========================================================
    # 範囲外
    # ========================================================

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

        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        continue

    # ========================================================
    # HSV
    # ========================================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ========================================================
    # 白色マスク
    # ========================================================

    mask = cv2.inRange(hsv, np.array([0, 0, MIN_V]), np.array([180, MAX_S, 255]))

    # ========================================================
    # 小さなノイズ除去
    # ========================================================

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ========================================================
    # 輪郭
    # ========================================================

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w > MAX_WIDTH:
            continue

        if h > MAX_HEIGHT:
            continue

        perimeter = cv2.arcLength(contour, True)

        circ = circularity(area, perimeter)

        if circ < MIN_CIRCULARITY:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # ====================================================
        # スコア
        # ====================================================

        score = 0

        # 円形
        score += circ * 40

        # 小さい物体を優先
        if 5 <= area <= 150:
            score += 20

        # 正方形に近い
        ratio = min(w, h) / max(w, h)

        score += ratio * 20

        # 前フレームの候補に近い場合
        if previous_candidates:

            nearest = min(
                distance(cx, cy, p["cx"], p["cy"]) for p in previous_candidates
            )

            if nearest < 100:
                score += 30

            elif nearest < 200:
                score += 10

        candidates.append(
            {
                "frame": frame_number,
                "cx": cx,
                "cy": cy,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "circularity": circ,
                "score": score,
            }
        )

    # ========================================================
    # スコア順
    # ========================================================

    candidates.sort(key=lambda x: x["score"], reverse=True)

    top_candidates = candidates[:TOP_N]

    # ========================================================
    # CSV保存
    # ========================================================

    for c in top_candidates:

        all_candidates.append(c)

    # ========================================================
    # 描画
    # ========================================================

    for rank, c in enumerate(top_candidates):

        x = int(c["x"])
        y = int(c["y"])
        w = int(c["w"])
        h = int(c["h"])

        cx = int(c["cx"])
        cy = int(c["cy"])

        # 上位候補を表示
        if rank == 0:

            color = (0, 0, 255)

        else:

            color = (0, 255, 255)

        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)

        cv2.circle(debug, (cx, cy), 4, color, -1)

        cv2.putText(
            debug,
            f"{rank + 1}:{c['score']:.1f}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    # ========================================================
    # 情報
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
        f"Candidates: {len(candidates)}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    if top_candidates:

        best = top_candidates[0]

        cv2.putText(
            debug,
            f"BEST ({best['cx']:.0f},{best['cy']:.0f})",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    out.write(debug)

    previous_candidates = top_candidates

    previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ============================================================
# 終了
# ============================================================

cap.release()
out.release()


# ============================================================
# CSV
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
            "circularity",
            "score",
        ]
    )

    for c in all_candidates:

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
                c["circularity"],
                c["score"],
            ]
        )


# ============================================================
# 結果
# ============================================================

print()
print("==========================================")
print("検出結果")
print("==========================================")

print("候補総数:", len(all_candidates))

print("CSV:", OUTPUT_CSV)

print("デバッグ動画:", OUTPUT_VIDEO)

print()
print("==========================================")
print("完了")
print("==========================================")
