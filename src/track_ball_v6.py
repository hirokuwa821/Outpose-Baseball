import cv2
import csv
import math
from pathlib import Path

from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================

MODEL_PATH = Path(r"runs\detect\runs\detect\ball_finetune_mydata_v2\weights\best.pt")

VIDEO_DIR = Path("videos")

OUTPUT_CSV_DIR = Path("output/ball_tracking_v6")
OUTPUT_VIDEO_DIR = Path("runs/detect/ball_tracking_v6")

OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


VIDEO_NAMES = [
    "pitch_102_1.mp4",
    "pitch_98_1.mp4",
    "pitch_98_2.mp4",
]


# ------------------------------------------------------------
# YOLO設定
# ------------------------------------------------------------

CONF_THRESHOLD = 0.15

# ボールの移動量として許容する最大値
MAX_MOVE_PER_FRAME = 80.0

# 検出が途切れても接続を試す最大フレーム数
MAX_GAP = 6

# 軌跡として最低限必要な実検出点数
MIN_REAL_POINTS = 5

# 軌跡の最低長
MIN_TRACK_LENGTH = 8

# ホーム付近まで追跡を延長したい目標フレーム
TARGET_END_FRAMES = {
    "pitch_102_1": 70,
    "pitch_98_1": 94,
    "pitch_98_2": 88,
}

# 最後の実検出から短距離だけ外挿
MAX_EXTRAPOLATION = 3
VELOCITY_HISTORY = 4


# ============================================================
# 距離
# ============================================================


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# ============================================================
# 検出候補
# ============================================================


def get_detections(model, frame):

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:

        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        w = x2 - x1
        h = y2 - y1

        detections.append(
            {
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "conf": conf,
            }
        )

    return detections


# ============================================================
# 軌跡生成
# ============================================================


def build_track(detections_by_frame):

    frames = sorted(detections_by_frame.keys())

    if not frames:
        return []

    tracks = []

    # --------------------------------------------------------
    # まず各検出を始点として追跡
    # --------------------------------------------------------

    for start_frame in frames:

        candidates = detections_by_frame[start_frame]

        for start_detection in candidates:

            track = [
                {
                    "frame": start_frame,
                    "cx": start_detection["cx"],
                    "cy": start_detection["cy"],
                    "conf": start_detection["conf"],
                    "interpolated": False,
                    "extrapolated": False,
                }
            ]

            current_frame = start_frame
            current_x = start_detection["cx"]
            current_y = start_detection["cy"]

            gap = 0

            while True:

                next_frame = current_frame + 1

                # 次の検出が存在しない
                if next_frame not in detections_by_frame:

                    gap += 1

                    if gap > MAX_GAP:
                        break

                    current_frame = next_frame

                    continue

                candidates_next = detections_by_frame[next_frame]

                best = None
                best_distance = float("inf")

                for candidate in candidates_next:

                    d = distance(
                        (current_x, current_y), (candidate["cx"], candidate["cy"])
                    )

                    if d <= MAX_MOVE_PER_FRAME:

                        if d < best_distance:

                            best_distance = d
                            best = candidate

                if best is None:

                    gap += 1

                    if gap > MAX_GAP:
                        break

                    current_frame = next_frame

                    continue

                # ------------------------------------------------
                # 実検出
                # ------------------------------------------------

                track.append(
                    {
                        "frame": next_frame,
                        "cx": best["cx"],
                        "cy": best["cy"],
                        "conf": best["conf"],
                        "interpolated": False,
                    }
                )

                current_x = best["cx"]
                current_y = best["cy"]

                current_frame = next_frame

                gap = 0

        if len(track) >= MIN_TRACK_LENGTH:

            real_points = sum(1 for p in track if not p["interpolated"])

            if real_points >= MIN_REAL_POINTS:

                tracks.append(track)

    if not tracks:
        return []

    # --------------------------------------------------------
    # 最も長い軌跡を基本採用
    # --------------------------------------------------------

    tracks.sort(
        key=lambda t: (len(t), sum(p["conf"] for p in t) / len(t)), reverse=True
    )

    return tracks[0]


# ============================================================
# 補間
# ============================================================


def interpolate_track(track):

    if not track:
        return track

    result = []

    for i in range(len(track) - 1):

        current = track[i]
        next_point = track[i + 1]

        result.append(current)

        frame_diff = next_point["frame"] - current["frame"]

        # 最大3フレームの欠損だけ補間
        if 1 < frame_diff <= MAX_GAP + 1:

            for j in range(1, frame_diff):

                ratio = j / frame_diff

                x = current["cx"] + (next_point["cx"] - current["cx"]) * ratio

                y = current["cy"] + (next_point["cy"] - current["cy"]) * ratio

                result.append(
                    {
                        "frame": current["frame"] + j,
                        "cx": x,
                        "cy": y,
                        "conf": 0.0,
                        "interpolated": True,
                        "extrapolated": False,
                    }
                )

    result.append(track[-1])

    result.sort(key=lambda p: p["frame"])

    return result


# ============================================================
# ホーム付近まで短距離外挿
# ============================================================


def extrapolate_to_target(track, target_frame):

    if not track or target_frame <= track[-1]["frame"]:
        return track

    gap = target_frame - track[-1]["frame"]

    if gap > MAX_EXTRAPOLATION:
        return track

    real_points = [
        p
        for p in track
        if not p.get("interpolated", False) and not p.get("extrapolated", False)
    ]

    if len(real_points) < 2:
        return track

    recent = real_points[-VELOCITY_HISTORY:]
    first = recent[0]
    last = recent[-1]

    dt = last["frame"] - first["frame"]

    if dt <= 0:
        return track

    vx = (last["cx"] - first["cx"]) / dt
    vy = (last["cy"] - first["cy"]) / dt

    step = math.sqrt(vx * vx + vy * vy)

    if step > MAX_MOVE_PER_FRAME:
        scale = MAX_MOVE_PER_FRAME / step
        vx *= scale
        vy *= scale

    result = list(track)

    for i in range(1, gap + 1):
        result.append(
            {
                "frame": last["frame"] + i,
                "cx": last["cx"] + vx * i,
                "cy": last["cy"] + vy * i,
                "conf": 0.0,
                "interpolated": False,
                "extrapolated": True,
            }
        )

    result.sort(key=lambda p: p["frame"])
    return result


# ============================================================
# 軌跡の評価
# ============================================================


def calculate_track_info(track):

    if len(track) < 2:
        return 0, 0

    total_distance = 0

    for i in range(len(track) - 1):

        p1 = track[i]
        p2 = track[i + 1]

        total_distance += distance((p1["cx"], p1["cy"]), (p2["cx"], p2["cy"]))

    real_points = [p for p in track if not p["interpolated"]]

    if real_points:

        mean_conf = sum(p["conf"] for p in real_points) / len(real_points)

    else:

        mean_conf = 0

    return total_distance, mean_conf


# ============================================================
# 動画処理
# ============================================================


def process_video(model, video_path):

    print()
    print("=" * 60)
    print("動画:", video_path)
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません:", video_path)

        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("サイズ:", width, "x", height)
    print("総フレーム数:", total_frames)

    # --------------------------------------------------------
    # 全フレーム検出
    # --------------------------------------------------------

    detections_by_frame = {}

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = get_detections(model, frame)

        if detections:

            detections_by_frame[frame_index] = detections

        frame_index += 1

    cap.release()

    print()
    print("------------------------------------------")
    print("検出結果")
    print("------------------------------------------")

    detection_count = sum(len(v) for v in detections_by_frame.values())

    print("検出フレーム数:", len(detections_by_frame))
    print("検出候補数:", detection_count)

    # --------------------------------------------------------
    # 軌跡
    # --------------------------------------------------------

    track = build_track(detections_by_frame)

    if not track:

        print()
        print("有効な軌跡が見つかりません")

        return

    # 補間
    track = interpolate_track(track)

    # ホーム付近まで短距離外挿
    video_key = video_path.stem
    target_frame = TARGET_END_FRAMES.get(video_key)

    if target_frame is not None:
        before_end = track[-1]["frame"]
        track = extrapolate_to_target(track, target_frame)
        after_end = track[-1]["frame"]

        if after_end > before_end:
            print()
            print(f"ホーム付近まで外挿: " f"Frame {before_end} → {after_end}")

    real_track = [
        p
        for p in track
        if not p.get("interpolated", False) and not p.get("extrapolated", False)
    ]

    total_distance, mean_conf = calculate_track_info(track)

    interpolated_count = sum(1 for p in track if p.get("interpolated", False))
    extrapolated_count = sum(1 for p in track if p.get("extrapolated", False))

    print()
    print("------------------------------------------")
    print("採用軌跡")
    print("------------------------------------------")

    print(f"Frame: " f"{track[0]['frame']} → " f"{track[-1]['frame']}")

    print("実検出点数:", len(real_track))

    print("追跡点数:", len(track))

    print(f"平均confidence: {mean_conf:.3f}")

    print(f"移動距離: {total_distance:.2f}px")

    print("補間点数:", interpolated_count)
    print("外挿点数:", extrapolated_count)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = OUTPUT_CSV_DIR / f"{video_path.stem}_tracking.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "time_sec",
                "cx",
                "cy",
                "confidence",
                "interpolated",
                "extrapolated",
            ]
        )

        for p in track:

            writer.writerow(
                [
                    p["frame"],
                    p["frame"] / fps,
                    f"{p['cx']:.3f}",
                    f"{p['cy']:.3f}",
                    f"{p['conf']:.4f}",
                    int(p.get("interpolated", False)),
                    int(p.get("extrapolated", False)),
                ]
            )

    # --------------------------------------------------------
    # 軌跡動画
    # --------------------------------------------------------

    output_video = OUTPUT_VIDEO_DIR / f"{video_path.stem}_tracking.mp4"

    cap = cv2.VideoCapture(str(video_path))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    track_by_frame = {p["frame"]: p for p in track}

    previous = None

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index in track_by_frame:

            p = track_by_frame[frame_index]

            cx = int(p["cx"])
            cy = int(p["cy"])

            # ------------------------------------------------
            # 補間点か実検出か
            # ------------------------------------------------

            if p.get("extrapolated", False):

                # 緑 = 外挿
                color = (0, 255, 0)

            elif p.get("interpolated", False):

                # 黄色 = 補間
                color = (0, 255, 255)

            else:

                # 赤 = 実検出
                color = (0, 0, 255)

            cv2.circle(frame, (cx, cy), 8, color, -1)

            cv2.circle(frame, (cx, cy), 14, color, 2)

            if previous is not None:

                cv2.line(frame, previous, (cx, cy), (255, 0, 0), 2)

            previous = (cx, cy)

            text = f"Frame {frame_index} " f"Conf {p['conf']:.2f}"

            if p.get("interpolated", False):

                text += " [INTERPOLATED]"

            elif p.get("extrapolated", False):

                text += " [EXTRAPOLATED]"

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        else:

            previous = None

        writer.write(frame)

        frame_index += 1

    cap.release()
    writer.release()

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("動画:")
    print(output_video)


# ============================================================
# メイン
# ============================================================


def main():

    print("=" * 60)
    print("ボール追跡 v6")
    print("実検出優先 + 最大6フレーム補間 + ホーム付近短距離外挿")
    print("=" * 60)

    print()
    print("モデル:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():

        print()
        print("ERROR: モデルがありません")
        print(MODEL_PATH)

        return

    model = YOLO(str(MODEL_PATH))

    print("モデル読み込み完了")

    for video_name in VIDEO_NAMES:

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():

            print()
            print("動画がありません:")
            print(video_path)

            continue

        process_video(model, video_path)

    print()
    print("=" * 60)
    print("全動画の追跡完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
