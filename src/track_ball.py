from ultralytics import YOLO
import cv2
import csv
import math
from pathlib import Path
import sys

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "runs/detect/train-2/weights/best.pt"

CONF = 0.01

# YOLO検出画像を拡大して検出
UPSCALE = 2.0

# ボール候補として許容するサイズ
MIN_BOX_SIZE = 2
MAX_BOX_SIZE = 150

# 検出候補動画
OUTPUT_VIDEO = "runs/detect/ball_debug.mp4"

# 全候補CSV
OUTPUT_CSV = "output/ball_candidates.csv"

# ============================================================
# 動画パス
# ============================================================

if len(sys.argv) < 2:
    print("使い方:")
    print("python src/track_ball.py videos\\pitch_102_1.mp4")
    sys.exit()

VIDEO_PATH = sys.argv[1]

print()
print("==========================================")
print("ボール追跡 v4")
print("==========================================")
print("入力動画:", VIDEO_PATH)
print("モデル:", MODEL_PATH)
print("CONF:", CONF)
print("UPSCALE:", UPSCALE)


# ============================================================
# モデル
# ============================================================

model = YOLO(MODEL_PATH)


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
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)


# ============================================================
# 出力先
# ============================================================

Path("output").mkdir(parents=True, exist_ok=True)
Path("runs/detect").mkdir(parents=True, exist_ok=True)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# ============================================================
# 全候補保存
# ============================================================

all_candidates = []

frame_number = 0


# ============================================================
# フレーム処理
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO用画像を拡大
    # --------------------------------------------------------

    if UPSCALE != 1.0:

        detect_frame = cv2.resize(
            frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC
        )

    else:

        detect_frame = frame

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = model.predict(detect_frame, conf=CONF, imgsz=1280, verbose=False)

    candidates = []

    # --------------------------------------------------------
    # 検出結果
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(float, box.xyxy[0])

            # 元画像サイズに戻す
            x1 /= UPSCALE
            y1 /= UPSCALE
            x2 /= UPSCALE
            y2 /= UPSCALE

            box_width = x2 - x1
            box_height = y2 - y1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # ------------------------------------------------
            # サイズフィルタ
            # ------------------------------------------------

            if box_width < MIN_BOX_SIZE:
                continue

            if box_height < MIN_BOX_SIZE:
                continue

            if box_width > MAX_BOX_SIZE:
                continue

            if box_height > MAX_BOX_SIZE:
                continue

            candidates.append(
                {
                    "frame": frame_number,
                    "confidence": confidence,
                    "cx": cx,
                    "cy": cy,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": box_width,
                    "height": box_height,
                }
            )

    # ========================================================
    # CSV用に保存
    # ========================================================

    for c in candidates:
        all_candidates.append(c)

    # ========================================================
    # デバッグ動画
    # ========================================================

    debug_frame = frame.copy()

    # --------------------------------------------------------
    # 全YOLO候補を表示
    # --------------------------------------------------------

    for c in candidates:

        x1 = int(c["x1"])
        y1 = int(c["y1"])
        x2 = int(c["x2"])
        y2 = int(c["y2"])

        cx = int(c["cx"])
        cy = int(c["cy"])

        conf = c["confidence"]

        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.circle(debug_frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            debug_frame,
            f"{conf:.3f}",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    # --------------------------------------------------------
    # フレーム情報
    # --------------------------------------------------------

    cv2.putText(
        debug_frame,
        f"Frame: {frame_number}/{total_frames}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        debug_frame,
        f"Candidates: {len(candidates)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # 投球前後のフレーム確認用
    # --------------------------------------------------------

    if 35 <= frame_number <= 80:

        cv2.rectangle(debug_frame, (0, 0), (width - 1, height - 1), (255, 0, 0), 3)

    out.write(debug_frame)

    # --------------------------------------------------------
    # ログ
    # --------------------------------------------------------

    if candidates:

        best = max(candidates, key=lambda x: x["confidence"])

        print(
            f"frame {frame_number:3d} | "
            f"detections={len(candidates):2d} | "
            f"best_conf={best['confidence']:.3f} | "
            f"center=({best['cx']:.1f},{best['cy']:.1f})"
        )


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
            "confidence",
            "center_x",
            "center_y",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
        ]
    )

    for c in all_candidates:

        writer.writerow(
            [
                c["frame"],
                c["frame"] / fps,
                c["confidence"],
                c["cx"],
                c["cy"],
                c["x1"],
                c["y1"],
                c["x2"],
                c["y2"],
                c["width"],
                c["height"],
            ]
        )


# ============================================================
# 結果
# ============================================================

print()
print("==========================================")
print("検出結果")
print("==========================================")

print("総フレーム数:", total_frames)
print("検出候補数:", len(all_candidates))


if all_candidates:

    best = max(all_candidates, key=lambda x: x["confidence"])

    print()
    print("最高confidence:")
    print(f"frame      : {best['frame']}")
    print(f"time       : {best['frame'] / fps:.3f} sec")
    print(f"confidence : {best['confidence']:.4f}")
    print(f"center     : " f"({best['cx']:.1f}, {best['cy']:.1f})")

    print()
    print("検出フレーム:")

    frames = sorted(set(c["frame"] for c in all_candidates))

    print(frames)


else:

    print()
    print("YOLOによる候補検出はありませんでした。")


print()
print("==========================================")
print("保存先")
print("==========================================")

print("デバッグ動画:")
print(OUTPUT_VIDEO)

print()
print("候補CSV:")
print(OUTPUT_CSV)

print()
print("==========================================")
print("処理完了")
print("==========================================")
