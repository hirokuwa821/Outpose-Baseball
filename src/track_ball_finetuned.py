from ultralytics import YOLO
import cv2
import math
import csv
from pathlib import Path

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "runs/detect/runs/detect/" "ball_finetune_mydata/weights/best.pt"

VIDEOS = [
    "videos/pitch_102_1.mp4",
    "videos/pitch_98_1.mp4",
    "videos/pitch_98_2.mp4",
]

OUTPUT_DIR = Path("output/finetuned_tracking")
VIDEO_DIR = Path("runs/detect/finetuned_tracking")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# 検出信頼度
CONF = 0.05

# 前フレームから許容する最大移動距離
MAX_DISTANCE = 120

# 最低連続フレーム数
MIN_TRACK_LENGTH = 5

# ============================================================
# モデル
# ============================================================

print("==========================================")
print("追加学習モデルによるボール追跡")
print("==========================================")

print("モデル:")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# 動画ごとの処理
# ============================================================

for video_path in VIDEOS:

    print()
    print("############################################################")
    print("動画:", video_path)
    print("############################################################")

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

    # --------------------------------------------------------
    # 全フレームの候補
    # --------------------------------------------------------

    frame_candidates = {}

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
                        "center_x": cx,
                        "center_y": cy,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        frame_candidates[frame_number] = candidates

    cap.release()

    # ========================================================
    # 追跡軌跡を作る
    # ========================================================

    tracks = []

    # 各フレームの候補から軌跡を延長する
    for frame in range(1, total_frames + 1):

        candidates = frame_candidates.get(frame, [])

        if not candidates:
            continue

        # ----------------------------------------------------
        # 既存軌跡へ接続
        # ----------------------------------------------------

        used_candidates = set()

        for track in tracks:

            if not track:
                continue

            last = track[-1]

            # 1フレーム前以外なら接続しない
            if frame != last["frame"] + 1:
                continue

            best_index = None
            best_distance = float("inf")

            for i, candidate in enumerate(candidates):

                if i in used_candidates:
                    continue

                dx = candidate["center_x"] - last["center_x"]

                dy = candidate["center_y"] - last["center_y"]

                distance = math.sqrt(dx * dx + dy * dy)

                if distance < best_distance:

                    best_distance = distance
                    best_index = i

            # ------------------------------------------------
            # 接続可能なら追加
            # ------------------------------------------------

            if best_index is not None and best_distance <= MAX_DISTANCE:

                track.append(candidates[best_index])

                used_candidates.add(best_index)

        # ----------------------------------------------------
        # どの軌跡にも入らなかった候補から新規軌跡
        # ----------------------------------------------------

        for i, candidate in enumerate(candidates):

            if i in used_candidates:
                continue

            tracks.append([candidate])

    # ========================================================
    # 軌跡評価
    # ========================================================

    valid_tracks = []

    for track in tracks:

        if len(track) < MIN_TRACK_LENGTH:
            continue

        # ----------------------------------------------------
        # 移動距離
        # ----------------------------------------------------

        total_distance = 0.0

        for i in range(1, len(track)):

            dx = track[i]["center_x"] - track[i - 1]["center_x"]

            dy = track[i]["center_y"] - track[i - 1]["center_y"]

            total_distance += math.sqrt(dx * dx + dy * dy)

        # ----------------------------------------------------
        # 始点・終点距離
        # ----------------------------------------------------

        dx = track[-1]["center_x"] - track[0]["center_x"]

        dy = track[-1]["center_y"] - track[0]["center_y"]

        straight_distance = math.sqrt(dx * dx + dy * dy)

        if total_distance > 0:

            straightness = straight_distance / total_distance

        else:

            straightness = 0

        # ----------------------------------------------------
        # 平均confidence
        # ----------------------------------------------------

        mean_confidence = sum(x["confidence"] for x in track) / len(track)

        # ----------------------------------------------------
        # 平均移動距離
        # ----------------------------------------------------

        if len(track) > 1:

            mean_move = total_distance / (len(track) - 1)

        else:

            mean_move = 0

        # ----------------------------------------------------
        # 評価値
        #
        # 長い軌跡
        # + 高confidence
        # + 適度な直進性
        # ----------------------------------------------------

        evaluation = len(track) * 10 + mean_confidence * 100 + straightness * 20

        valid_tracks.append(
            {
                "track": track,
                "length": len(track),
                "total_distance": total_distance,
                "straightness": straightness,
                "mean_confidence": mean_confidence,
                "mean_move": mean_move,
                "evaluation": evaluation,
            }
        )

    # ========================================================
    # TOP 20
    # ========================================================

    valid_tracks.sort(key=lambda x: x["evaluation"], reverse=True)

    print()
    print("==========================================")
    print("有力軌跡 TOP 20")
    print("==========================================")

    if not valid_tracks:

        print("有力な軌跡がありません")
        continue

    for i, result in enumerate(valid_tracks[:20], start=1):

        track = result["track"]

        start = track[0]
        end = track[-1]

        print()
        print(f"[{i}]")
        print(f"Frame {start['frame']} → " f"{end['frame']}")

        print("length       :", result["length"])

        print(
            "start        : " f"({start['center_x']:.1f}, " f"{start['center_y']:.1f})"
        )

        print("end          : " f"({end['center_x']:.1f}, " f"{end['center_y']:.1f})")

        print("distance     :", f"{result['total_distance']:.2f}px")

        print("straightness :", f"{result['straightness']:.3f}")

        print("mean move    :", f"{result['mean_move']:.2f}px")

        print("mean conf    :", f"{result['mean_confidence']:.3f}")

        print("evaluation   :", f"{result['evaluation']:.2f}")

    # ========================================================
    # 1位軌跡を保存
    # ========================================================

    best = valid_tracks[0]
    best_track = best["track"]

    video_name = Path(video_path).stem

    csv_path = OUTPUT_DIR / f"{video_name}_tracking.csv"

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

        for d in best_track:

            writer.writerow(
                [
                    d["frame"],
                    d["frame"] / fps,
                    d["center_x"],
                    d["center_y"],
                    d["confidence"],
                ]
            )

    # ========================================================
    # 追跡結果動画
    # ========================================================

    output_video = VIDEO_DIR / f"{video_name}_tracking.mp4"

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    best_frames = {d["frame"]: d for d in best_track}

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number in best_frames:

            d = best_frames[frame_number]

            x1 = d["x1"]
            y1 = d["y1"]
            x2 = d["x2"]
            y2 = d["y2"]

            cx = int(d["center_x"])
            cy = int(d["center_y"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"BALL {d['confidence']:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame, "TRACK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
            )

        writer.write(frame)

    cap.release()
    writer.release()

    # ========================================================
    # 結果
    # ========================================================

    print()
    print("==========================================")
    print("採用軌跡")
    print("==========================================")

    print("Frame:", best_track[0]["frame"], "→", best_track[-1]["frame"])

    print("追跡点数:", len(best_track))

    print("平均confidence:", f"{best['mean_confidence']:.3f}")

    print("移動距離:", f"{best['total_distance']:.2f}px")

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("動画:")
    print(output_video)


print()
print("==========================================")
print("全動画の追跡完了")
print("==========================================")
