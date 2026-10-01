import cv2
import csv
from pathlib import Path
from collections import defaultdict

# ============================================================
# 設定
# ============================================================

VIDEO_PATH = "videos/pitch_102_1.mp4"

INPUT_CSV = "output/ball_motion_candidates_v2.csv"

OUTPUT_VIDEO = "runs/detect/ball_candidate_visualization.mp4"

START_FRAME = 35
END_FRAME = 80

# 表示する候補数
MAX_DISPLAY = 10


# ============================================================
# CSV読み込み
# ============================================================

candidates = defaultdict(list)

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        frame = int(row["frame"])

        if frame < START_FRAME or frame > END_FRAME:
            continue

        candidates[frame].append(
            {
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "score": float(row["score"]),
            }
        )


# ============================================================
# 動画
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print()
print("==========================================")
print("候補可視化")
print("==========================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)

print("表示範囲:", START_FRAME, "～", END_FRAME)


# ============================================================
# 出力
# ============================================================

Path("runs/detect").mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# ============================================================
# フレーム処理
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # ----------------------------------------
    # 対象範囲以外
    # ----------------------------------------

    if frame_number < START_FRAME or frame_number > END_FRAME:

        writer.write(frame)

        continue

    frame_candidates = candidates.get(frame_number, [])

    # ========================================================
    # score順に並べる
    # ========================================================

    frame_candidates = sorted(frame_candidates, key=lambda x: x["score"], reverse=True)

    # 上位だけ表示
    frame_candidates = frame_candidates[:MAX_DISPLAY]

    # ========================================================
    # 候補を描画
    # ========================================================

    for rank, c in enumerate(frame_candidates, 1):

        x = int(c["x"])
        y = int(c["y"])
        score = c["score"]

        # ----------------------------------------
        # 順位によって大きさを変える
        # ----------------------------------------

        if rank == 1:

            radius = 14
            thickness = 4

        else:

            radius = 8
            thickness = 2

        # ----------------------------------------
        # 候補点
        # ----------------------------------------

        cv2.circle(frame, (x, y), radius, (0, 0, 255), thickness)

        # ----------------------------------------
        # 順位
        # ----------------------------------------

        cv2.putText(
            frame,
            f"{rank}:{score:.3f}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    # ========================================================
    # フレーム情報
    # ========================================================

    cv2.putText(
        frame,
        f"FRAME {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"CANDIDATES: {len(candidates.get(frame_number, []))}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # ========================================================
    # 出力
    # ========================================================

    writer.write(frame)


# ============================================================
# 終了
# ============================================================

cap.release()
writer.release()


print()
print("==========================================")
print("完了")
print("==========================================")

print("出力動画:", OUTPUT_VIDEO)
