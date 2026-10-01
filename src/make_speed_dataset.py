import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import re

# ==========================================
# 設定
# ==========================================

MODEL_PATH = "runs/detect/train-2/weights/best.pt"
VIDEO_DIR = Path("test_videos")
OUTPUT_FILE = Path("output/speed_dataset.csv")

# 球速を動画ファイル名から取得
# 例: mlb_001_81.1mph_xxxxx.mp4
VIDEOS = [
    "mlb_001_81.1mph_CGI1SSOSP466_right.mp4",
    "mlb_001_81.5mph_KPY8PN4UY84W_right.mp4",
    "mlb_001_82.5mph_HQCPRB2AYCFZ_right.mp4",
    "mlb_002_81.9mph_MM9QJ9EPUPHF_right.mp4",
]

# ==========================================
# YOLOモデル
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# 1動画を解析
# ==========================================


def analyze_video(video_path):

    print()
    print("=" * 50)
    print("解析中:", video_path.name)
    print("=" * 50)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("FPSを取得できませんでした")
        cap.release()
        return None

    detections = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # YOLOでボール検出
        results = model.predict(frame, conf=0.25, verbose=False)

        candidates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = xyxy

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                confidence = float(box.conf[0])

                candidates.append((center_x, center_y, confidence))

        # ----------------------------------
        # 前フレームとの距離が近いものを選択
        # ----------------------------------

        if candidates:

            if not detections:

                # 最初の検出
                best = max(candidates, key=lambda x: x[2])

            else:

                previous_x = detections[-1]["center_x"]
                previous_y = detections[-1]["center_y"]

                best = min(
                    candidates,
                    key=lambda x: (x[0] - previous_x) ** 2 + (x[1] - previous_y) ** 2,
                )

                distance = np.sqrt(
                    (best[0] - previous_x) ** 2 + (best[1] - previous_y) ** 2
                )

                # あまりにも離れていたら無視
                if distance > 100:
                    continue

            detections.append(
                {
                    "frame": frame_number,
                    "time_sec": frame_number / fps,
                    "center_x": best[0],
                    "center_y": best[1],
                    "confidence": best[2],
                }
            )

    cap.release()

    # ======================================
    # 連続検出区間を探す
    # ======================================

    if len(detections) < 5:
        print("検出数が少なすぎます")
        return None

    segments = []

    current = [detections[0]]

    for i in range(1, len(detections)):

        frame_gap = detections[i]["frame"] - detections[i - 1]["frame"]

        if frame_gap <= 1:

            current.append(detections[i])

        else:

            if len(current) >= 5:
                segments.append(current)

            current = [detections[i]]

    if len(current) >= 5:
        segments.append(current)

    if not segments:
        print("連続検出区間がありません")
        return None

    # 一番長い区間を投球として採用
    pitch = max(segments, key=len)

    print("投球区間:", pitch[0]["frame"], "～", pitch[-1]["frame"])

    # ======================================
    # DataFrame化
    # ======================================

    df = pd.DataFrame(pitch)

    x = df["center_x"].to_numpy()
    y = df["center_y"].to_numpy()
    t = df["time_sec"].to_numpy()

    # ======================================
    # 特徴量
    # ======================================

    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(dx**2 + dy**2)

    dt = np.diff(t)

    # 0除算防止
    valid = dt > 0

    pixel_speeds = np.zeros_like(distances)

    pixel_speeds[valid] = distances[valid] / dt[valid]

    total_distance = distances.sum()

    duration = t[-1] - t[0]

    if duration > 0:

        average_pixel_speed = total_distance / duration

    else:

        average_pixel_speed = 0

    max_pixel_speed = pixel_speeds.max() if len(pixel_speeds) > 0 else 0

    start_end_distance = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

    # 方向変化
    direction_change = 0

    if len(dx) >= 2:

        angles = np.arctan2(dy, dx)

        angle_diff = np.diff(angles)

        angle_diff = np.abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

        direction_change = angle_diff.sum()

    # ======================================
    # 結果
    # ======================================

    features = {
        "frame_start": pitch[0]["frame"],
        "frame_end": pitch[-1]["frame"],
        "num_frames": len(pitch),
        "duration": duration,
        "total_distance_px": total_distance,
        "start_end_distance_px": start_end_distance,
        "average_pixel_speed": average_pixel_speed,
        "max_pixel_speed": max_pixel_speed,
        "total_dx": x[-1] - x[0],
        "total_dy": y[-1] - y[0],
        "direction_change": direction_change,
        "mean_confidence": df["confidence"].mean(),
        "min_confidence": df["confidence"].min(),
    }

    return features


# ==========================================
# 球速取得
# ==========================================


def get_speed(video_name):

    match = re.search(r"(\d+(?:\.\d+)?)mph", video_name)

    if not match:
        return None

    speed_mph = float(match.group(1))

    speed_kmh = speed_mph * 1.609344

    return speed_mph, speed_kmh


# ==========================================
# 全動画解析
# ==========================================

all_rows = []

for video_name in VIDEOS:

    video_path = VIDEO_DIR / video_name

    if not video_path.exists():

        print("動画が見つかりません:", video_path)

        continue

    features = analyze_video(video_path)

    if features is None:
        continue

    speed = get_speed(video_name)

    if speed is None:

        print("球速を取得できません:", video_name)

        continue

    speed_mph, speed_kmh = speed

    features["video_name"] = video_name

    features["speed_mph"] = speed_mph

    features["speed_kmh"] = speed_kmh

    all_rows.append(features)


# ==========================================
# 保存
# ==========================================

if not all_rows:

    print("データが作成できませんでした")

else:

    dataset = pd.DataFrame(all_rows)

    columns = [
        "video_name",
        "speed_mph",
        "speed_kmh",
        "frame_start",
        "frame_end",
        "num_frames",
        "duration",
        "total_distance_px",
        "start_end_distance_px",
        "average_pixel_speed",
        "max_pixel_speed",
        "total_dx",
        "total_dy",
        "direction_change",
        "mean_confidence",
        "min_confidence",
    ]

    dataset = dataset[columns]

    dataset.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 50)
    print("球速データセット作成完了")
    print("=" * 50)

    print(dataset.to_string(index=False))

    print()
    print("保存先:")
    print(OUTPUT_FILE)
