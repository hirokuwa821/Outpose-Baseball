from ultralytics import YOLO
import cv2
from pathlib import Path
import math
import csv

# ==========================================
# 設定
# ==========================================

MODEL_PATH = "runs/detect/train-8/weights/best.pt"

VIDEO_DIR = Path("test_videos")

OUTPUT_VIDEO_DIR = Path("runs/detect/ball_tracks_v2")
OUTPUT_CSV_DIR = Path("output/ball_tracking_v2")

CONF = 0.20

# 最低連続フレーム数
MIN_CONSECUTIVE = 5

# 1フレームで許容する最大移動距離
MAX_DISTANCE = 100

# ボールを見失っても追跡を維持する最大フレーム数
MAX_MISSED_FRAMES = 5


# ==========================================
# YOLOモデル
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# 出力フォルダ
# ==========================================

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 対象動画
# ==========================================

CANDIDATE_CSV = Path("output/ball_label_candidates_20.csv")

video_paths = []

with open(CANDIDATE_CSV, "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        video_name = row["video"]

        video_path = VIDEO_DIR / video_name

        video_paths.append(video_path)


print()
print("対象動画数:", len(video_paths))


# ==========================================
# 1本ずつ処理
# ==========================================

for video_path in video_paths:

    print()
    print("==================================================")
    print("解析中:", video_path.name)
    print("==================================================")

    if not video_path.exists():
        print("動画がありません:", video_path)
        continue

    # ------------------------------------------
    # 動画
    # ------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

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

    # ------------------------------------------
    # 検出結果
    # ------------------------------------------

    detections = []

    previous_center = None
    previous_velocity = None

    missed_frames = 0

    frame_number = 0

    # ==========================================
    # YOLO検出
    # ==========================================

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

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                candidates.append(
                    {
                        "confidence": confidence,
                        "cx": cx,
                        "cy": cy,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        # ======================================
        # 候補選択
        # ======================================

        selected = None

        if candidates:

            # ----------------------------------
            # 初回検出
            # ----------------------------------

            if previous_center is None:

                selected = max(candidates, key=lambda x: x["confidence"])

            # ----------------------------------
            # 追跡中
            # ----------------------------------

            else:

                scored_candidates = []

                for candidate in candidates:

                    dx = candidate["cx"] - previous_center[0]

                    dy = candidate["cy"] - previous_center[1]

                    distance = math.sqrt(dx**2 + dy**2)

                    # 距離が大きすぎる候補は除外
                    if distance > MAX_DISTANCE:
                        continue

                    # ----------------------------------
                    # スコア
                    # ----------------------------------

                    score = distance

                    # 前回の移動方向が分かる場合
                    if previous_velocity is not None:

                        vx, vy = previous_velocity

                        velocity_length = math.sqrt(vx**2 + vy**2)

                        direction_length = math.sqrt(dx**2 + dy**2)

                        if velocity_length > 0 and direction_length > 0:

                            dot = vx * dx + vy * dy

                            cosine = dot / (velocity_length * direction_length)

                            # 同じ方向ならスコアを少し下げる
                            score -= cosine * 30

                    # 信頼度が高いほど有利
                    score -= candidate["confidence"] * 20

                    scored_candidates.append((score, candidate))

                if scored_candidates:

                    scored_candidates.sort(key=lambda x: x[0])

                    selected = scored_candidates[0][1]

        # ======================================
        # 検出成功
        # ======================================

        if selected is not None:

            current_center = (selected["cx"], selected["cy"])

            # ----------------------------------
            # 速度ベクトル
            # ----------------------------------

            if previous_center is not None:

                velocity = (
                    current_center[0] - previous_center[0],
                    current_center[1] - previous_center[1],
                )

                previous_velocity = velocity

            previous_center = current_center

            missed_frames = 0

            detections.append({"frame": frame_number, **selected})

        # ======================================
        # 検出失敗
        # ======================================

        else:

            missed_frames += 1

            print(
                f"  フレーム {frame_number}: "
                f"ボール未検出 "
                f"({missed_frames}/{MAX_MISSED_FRAMES})"
            )

            # ----------------------------------
            # 数フレームなら追跡を維持
            # ----------------------------------

            if missed_frames > MAX_MISSED_FRAMES:

                previous_center = None
                previous_velocity = None
                missed_frames = 0

    cap.release()

    # ==========================================
    # 連続検出区間
    # ==========================================

    segments = []

    current_segment = []

    previous_frame = None

    for detection in detections:

        frame = detection["frame"]

        if previous_frame is None:

            current_segment = [detection]

        elif frame == previous_frame + 1:

            current_segment.append(detection)

        else:

            if len(current_segment) >= MIN_CONSECUTIVE:

                segments.append(current_segment)

            current_segment = [detection]

        previous_frame = frame

    if len(current_segment) >= MIN_CONSECUTIVE:

        segments.append(current_segment)

    # ==========================================
    # 結果
    # ==========================================

    if not segments:

        print("連続検出区間がありません")
        continue

    print()
    print("連続検出区間")

    for i, segment in enumerate(segments):

        start = segment[0]["frame"]
        end = segment[-1]["frame"]

        print(
            f"区間 {i + 1}: " f"フレーム {start} ～ {end} " f"({len(segment)}フレーム)"
        )

    # ==========================================
    # 最長区間
    # ==========================================

    pitch_segment = max(segments, key=len)

    pitch_start = pitch_segment[0]["frame"]
    pitch_end = pitch_segment[-1]["frame"]

    print()
    print("投球候補")
    print("開始フレーム:", pitch_start)
    print("終了フレーム:", pitch_end)
    print("フレーム数:", len(pitch_segment))

    # ==========================================
    # ファイル名
    # ==========================================

    stem = video_path.stem

    output_csv = OUTPUT_CSV_DIR / f"{stem}_tracking.csv"

    output_video = OUTPUT_VIDEO_DIR / f"{stem}_tracking.mp4"

    # ==========================================
    # CSV保存
    # ==========================================

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow(["frame", "time_sec", "center_x", "center_y", "confidence"])

        for d in pitch_segment:

            frame = d["frame"]

            time_sec = frame / fps

            writer.writerow([frame, time_sec, d["cx"], d["cy"], d["confidence"]])

    print("CSV保存:", output_csv)

    # ==========================================
    # 投球区間動画
    # ==========================================

    pitch_frames = {d["frame"]: d for d in pitch_segment}

    cap = cv2.VideoCapture(str(video_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number in pitch_frames:

            d = pitch_frames[frame_number]

            x1 = d["x1"]
            y1 = d["y1"]
            x2 = d["x2"]
            y2 = d["y2"]

            cx = d["cx"]
            cy = d["cy"]

            confidence = d["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

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
                frame, "PITCH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
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
                "NOT PITCH",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        out.write(frame)

    cap.release()
    out.release()

    print("動画保存:", output_video)


# ==========================================
# 完了
# ==========================================

print()
print("==================================================")
print(f"{len(video_paths)}本の解析が完了しました")
print("==================================================")
