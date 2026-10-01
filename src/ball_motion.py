import cv2
import numpy as np
from pathlib import Path

# =========================
# 設定
# =========================

VIDEO_PATH = Path("test_videos/mlb_001_81.1mph_CGI1SSOSP466_right.mp4")

OUTPUT_PATH = Path("output/ball_motion.mp4")

# 投球中を調べるフレーム範囲
START_FRAME = 120
END_FRAME = 170

# ボール候補の大きさ
MIN_AREA = 2
MAX_AREA = 150

# 前フレームとの差分の閾値
DIFF_THRESHOLD = 25


# =========================
# 動画を開く
# =========================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("サイズ:", width, "x", height)

# 出力動画
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))


# =========================
# 前フレーム
# =========================

previous_gray = None

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # 指定区間より前
    if frame_number < START_FRAME:
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue

    # 指定区間終了
    if frame_number > END_FRAME:
        break

    # グレースケール
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 最初のフレーム
    if previous_gray is None:
        previous_gray = gray
        continue

    # =========================
    # フレーム差分
    # =========================

    diff = cv2.absdiff(previous_gray, gray)

    # 二値化
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # =========================
    # ボール候補
    # =========================

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 小さな物体
        if w > 30 or h > 30:
            continue

        # 極端に細長いものを除外
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)

        if aspect > 5:
            continue

        center_x = x + w // 2
        center_y = y + h // 2

        candidates.append((center_x, center_y, w, h, area))

    # =========================
    # 描画
    # =========================

    result = frame.copy()

    # 投球区間を囲む
    cv2.rectangle(result, (150, 80), (600, 320), (255, 255, 0), 1)

    # 候補を表示
    for i, candidate in enumerate(candidates):

        cx, cy, w, h, area = candidate

        cv2.circle(result, (cx, cy), 5, (0, 0, 255), 2)

        cv2.putText(
            result,
            f"{i}:{area:.0f}",
            (cx + 5, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    # フレーム番号
    cv2.putText(
        result,
        f"Frame: {frame_number}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 候補数
    cv2.putText(
        result,
        f"Candidates: {len(candidates)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    out.write(result)

    # =========================
    # 座標を表示
    # =========================

    if candidates:

        print(f"frame {frame_number}: " f"{len(candidates)} candidates")

        for c in candidates:

            cx, cy, w, h, area = c

            print(f"  x={cx}, " f"y={cy}, " f"size={w}x{h}, " f"area={area:.1f}")

    previous_gray = gray


# =========================
# 終了
# =========================

cap.release()
out.release()

print()
print("解析終了")
print("出力動画:", OUTPUT_PATH)
