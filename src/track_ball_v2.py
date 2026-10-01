import cv2
import csv
import math
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "runs/detect/runs/detect/ball_finetune_mydata_v2/weights/best.pt"

VIDEOS = [
    "videos/pitch_102_1.mp4",
    "videos/pitch_98_1.mp4",
    "videos/pitch_98_2.mp4",
]

OUTPUT_DIR = Path("runs/detect/ball_tracking_v2")
CSV_DIR = Path("output/ball_tracking_v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# 検出信頼度
CONF = 0.20

# 投球らしい連続検出の最低フレーム数
MIN_SEGMENT_LENGTH = 5

# 前後フレーム間で許容する最大移動距離
MAX_MOVE = 150

# ============================================================
# モデル
# ============================================================

print("=" * 60)
print("v2モデル ボール追跡")
print("=" * 60)

print("モデル:")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# 動画ごとの処理
# ============================================================

for video_path in VIDEOS:

    print()
    print("#" * 60)
    print("動画:", video_path)
    print("#" * 60)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けません")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("サイズ:", width, "x", height)
    print("総フレーム数:", total_frames)

    # ========================================================
    # 全検出結果
    # ========================================================

    detections = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(frame, conf=CONF, verbose=False)

        candidates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

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
                    }
                )

        # そのフレームで最もconfidenceが高いものだけ採用
        if candidates:

            best = max(candidates, key=lambda x: x["confidence"])

            detections.append(best)

    cap.release()

    # ========================================================
    # 検出結果
    # ========================================================

    print()
    print("------------------------------------------")
    print("検出結果")
    print("------------------------------------------")

    print("検出フレーム数:", len(detections))

    if not detections:

        print("検出なし")
        continue

    print("最高confidence:", f"{max(d['confidence'] for d in detections):.4f}")

    # ========================================================
    # 連続検出区間を作る
    # ========================================================

    segments = []

    current = []

    previous = None

    for detection in detections:

        if previous is None:

            current = [detection]

        else:

            frame_gap = detection["frame"] - previous["frame"]

            distance = math.sqrt(
                (detection["cx"] - previous["cx"]) ** 2
                + (detection["cy"] - previous["cy"]) ** 2
            )

            # ------------------------------------------------
            # 連続していると判断
            # ------------------------------------------------

            if frame_gap <= 1 and distance <= MAX_MOVE:

                current.append(detection)

            else:

                if len(current) >= MIN_SEGMENT_LENGTH:
                    segments.append(current)

                current = [detection]

        previous = detection

    if len(current) >= MIN_SEGMENT_LENGTH:
        segments.append(current)

    # ========================================================
    # 区間表示
    # ========================================================

    print()
    print("------------------------------------------")
    print("連続検出区間")
    print("------------------------------------------")

    if not segments:

        print("連続検出区間なし")
        continue

    for i, segment in enumerate(segments):

        start = segment[0]["frame"]
        end = segment[-1]["frame"]

        mean_conf = sum(d["confidence"] for d in segment) / len(segment)

        distance = 0

        for j in range(1, len(segment)):

            distance += math.sqrt(
                (segment[j]["cx"] - segment[j - 1]["cx"]) ** 2
                + (segment[j]["cy"] - segment[j - 1]["cy"]) ** 2
            )

        print(
            f"[{i + 1}] "
            f"Frame {start} → {end} | "
            f"length={len(segment)} | "
            f"mean_conf={mean_conf:.3f} | "
            f"distance={distance:.1f}px"
        )

    # ========================================================
    # 投球区間の選択
    # ========================================================

    # 長さを優先し、その中でconfidenceを考慮
    best_segment = max(
        segments, key=lambda s: (len(s), sum(d["confidence"] for d in s) / len(s))
    )

    pitch_start = best_segment[0]["frame"]
    pitch_end = best_segment[-1]["frame"]

    mean_conf = sum(d["confidence"] for d in best_segment) / len(best_segment)

    # ========================================================
    # 軌跡距離
    # ========================================================

    total_distance = 0

    for i in range(1, len(best_segment)):

        total_distance += math.sqrt(
            (best_segment[i]["cx"] - best_segment[i - 1]["cx"]) ** 2
            + (best_segment[i]["cy"] - best_segment[i - 1]["cy"]) ** 2
        )

    # ========================================================
    # 採用結果
    # ========================================================

    print()
    print("==========================================")
    print("採用軌跡")
    print("==========================================")

    print(f"Frame: {pitch_start} → {pitch_end}")

    print("追跡点数:", len(best_segment))

    print("平均confidence:", f"{mean_conf:.3f}")

    print("移動距離:", f"{total_distance:.2f}px")

    # ========================================================
    # CSV
    # ========================================================

    video_name = Path(video_path).stem

    csv_path = CSV_DIR / f"{video_name}_tracking.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "time_sec",
                "center_x",
                "center_y",
                "confidence",
            ]
        )

        for d in best_segment:

            writer.writerow(
                [
                    d["frame"],
                    d["frame"] / fps,
                    d["cx"],
                    d["cy"],
                    d["confidence"],
                ]
            )

    print()
    print("CSV:")
    print(csv_path)

    # ========================================================
    # 結果動画
    # ========================================================

    cap = cv2.VideoCapture(video_path)

    output_video = OUTPUT_DIR / f"{video_name}_tracking.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    pitch_frames = {d["frame"]: d for d in best_segment}

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # 採用軌跡
        # ----------------------------------------------------

        if frame_number in pitch_frames:

            d = pitch_frames[frame_number]

            x1 = d["x1"]
            y1 = d["y1"]
            x2 = d["x2"]
            y2 = d["y2"]

            cx = int(d["cx"])
            cy = int(d["cy"])

            confidence = d["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"BALL {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "TRACKING",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"frame: {frame_number}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        else:

            cv2.putText(
                frame,
                "NO BALL",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        out.write(frame)

    cap.release()
    out.release()

    print()
    print("動画:")
    print(output_video)


print()
print("============================================================")
print("全動画の追跡完了")
print("============================================================")
