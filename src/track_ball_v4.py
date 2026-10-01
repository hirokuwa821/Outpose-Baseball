import cv2
import csv
import math
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/runs/detect/ball_finetune_mydata_v2/weights/best.pt")

VIDEO_LIST = [
    Path("videos/pitch_102_1.mp4"),
    Path("videos/pitch_98_1.mp4"),
    Path("videos/pitch_98_2.mp4"),
]

CSV_DIR = Path("output/ball_tracking_v4")
VIDEO_DIR = Path("runs/detect/ball_tracking_v4")
CSV_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.20
MAX_GAP = 3
MIN_REAL_POINTS = 5
MAX_MOVE = 80.0


def center_of_box(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def choose_detection(detections, previous_center):
    if not detections:
        return None

    best = None
    best_distance = float("inf")

    for det in detections:
        d = distance(det["center"], previous_center)
        if d < best_distance:
            best_distance = d
            best = det

    if best is None or best_distance > MAX_MOVE:
        return None

    return best


def interpolate(p1, p2, frame):
    ratio = (frame - p1["frame"]) / (p2["frame"] - p1["frame"])
    x = p1["x"] + (p2["x"] - p1["x"]) * ratio
    y = p1["y"] + (p2["y"] - p1["y"]) * ratio
    return x, y


def track_video(model, video_path):
    print()
    print("=" * 60)
    print("動画:", video_path)
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("動画を開けません:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("総フレーム数:", total_frames)

    detections_by_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                detections.append(
                    {
                        "center": center_of_box(xyxy),
                        "conf": conf,
                        "xyxy": xyxy,
                    }
                )

        detections_by_frame.append(detections)

    cap.release()

    # --------------------------------------------------------
    # 実検出だけで連続軌跡候補を作る
    # --------------------------------------------------------

    tracks = []

    for start_frame in range(total_frames):
        detections = detections_by_frame[start_frame]

        if not detections:
            continue

        first = max(detections, key=lambda d: d["conf"])

        points = [
            {
                "frame": start_frame,
                "x": first["center"][0],
                "y": first["center"][1],
                "conf": first["conf"],
                "source": "detected",
                "xyxy": first["xyxy"],
            }
        ]

        previous_center = first["center"]

        for current_frame in range(start_frame + 1, total_frames):
            candidate = choose_detection(
                detections_by_frame[current_frame], previous_center
            )

            if candidate is None:
                break

            points.append(
                {
                    "frame": current_frame,
                    "x": candidate["center"][0],
                    "y": candidate["center"][1],
                    "conf": candidate["conf"],
                    "source": "detected",
                    "xyxy": candidate["xyxy"],
                }
            )

            previous_center = candidate["center"]

        if len(points) >= MIN_REAL_POINTS:
            tracks.append(points)

    if not tracks:
        print("実検出だけでは有効な軌跡がありません")
        return

    # 最も長い実検出軌跡を採用
    tracks.sort(
        key=lambda p: (len(p), sum(x["conf"] for x in p) / len(p)), reverse=True
    )

    real_points = tracks[0]

    mean_conf = sum(p["conf"] for p in real_points) / len(real_points)

    direct_distance = distance(
        (real_points[0]["x"], real_points[0]["y"]),
        (real_points[-1]["x"], real_points[-1]["y"]),
    )

    # --------------------------------------------------------
    # 実検出点の「間」だけ補間
    # --------------------------------------------------------

    final_points = []
    interpolated_count = 0

    for i in range(len(real_points) - 1):
        p1 = real_points[i]
        p2 = real_points[i + 1]

        final_points.append(p1)

        gap = p2["frame"] - p1["frame"] - 1

        if 1 <= gap <= MAX_GAP:
            for frame in range(p1["frame"] + 1, p2["frame"]):
                x, y = interpolate(p1, p2, frame)

                final_points.append(
                    {
                        "frame": frame,
                        "x": x,
                        "y": y,
                        "conf": 0.0,
                        "source": "interpolated",
                        "xyxy": None,
                    }
                )

                interpolated_count += 1

    final_points.append(real_points[-1])

    print()
    print("------------------------------------------")
    print("採用軌跡")
    print("------------------------------------------")
    print(f"Frame: {real_points[0]['frame']} → " f"{real_points[-1]['frame']}")
    print(f"実検出点数: {len(real_points)}")
    print(f"追跡点数: {len(final_points)}")
    print(f"平均confidence: {mean_conf:.3f}")
    print(f"移動距離: {direct_distance:.2f}px")
    print(f"補間点数: {interpolated_count}")

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    csv_path = CSV_DIR / f"{video_path.stem}_tracking.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "x",
                "y",
                "confidence",
                "source",
            ]
        )

        for p in final_points:
            writer.writerow(
                [
                    p["frame"],
                    f"{p['x']:.3f}",
                    f"{p['y']:.3f}",
                    f"{p['conf']:.6f}",
                    p["source"],
                ]
            )

    # --------------------------------------------------------
    # 結果動画
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = VIDEO_DIR / f"{video_path.stem}_tracking.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    point_map = {p["frame"]: p for p in final_points}

    history = []
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index in point_map:
            p = point_map[frame_index]

            x = int(round(p["x"]))
            y = int(round(p["y"]))

            # 実検出=赤、補間=黄色
            color = (0, 0, 255) if p["source"] == "detected" else (0, 255, 255)

            cv2.circle(frame, (x, y), 7, color, -1)

            history.append((x, y))

        for a, b in zip(history[:-1], history[1:]):
            cv2.line(frame, a, b, (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Frame: {frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print()
    print("CSV:")
    print(csv_path)
    print()
    print("動画:")
    print(output_path)


def main():
    print("=" * 60)
    print("ボール追跡 v4")
    print("実検出を基本に、検出間の最大3フレームだけ線形補間")
    print("=" * 60)

    print()
    print("モデル:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        print("モデルがありません")
        return

    model = YOLO(str(MODEL_PATH))
    print("モデル読み込み完了")

    for video_path in VIDEO_LIST:
        if not video_path.exists():
            print("動画がありません:", video_path)
            continue

        track_video(model, video_path)

    print()
    print("=" * 60)
    print("全動画の追跡完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
