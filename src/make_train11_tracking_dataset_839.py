import csv
import math
import re
from pathlib import Path

import numpy as np

# ============================================================
# 設定
# ============================================================

TRACKING_DIR = Path("output/train11_tracking_839")

OUTPUT_CSV = Path("output/ball_tracking_dataset_train11_839.csv")


# ============================================================
# ファイル名から球速を取得
# ============================================================


def extract_speed_kmh(video_name):

    match = re.search(r"_(\d+(?:\.\d+)?)mph", video_name)

    if match is None:
        return None

    speed_mph = float(match.group(1))

    return speed_mph * 1.609344


# ============================================================
# 特徴量計算
# ============================================================


def calculate_features(csv_path):

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time_sec": float(row["time_sec"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "confidence": float(row["confidence"]),
                }
            )

    # 2点未満では計算できない
    if len(rows) < 2:
        return None

    # ========================================================
    # 基本情報
    # ========================================================

    frame_start = rows[0]["frame"]
    frame_end = rows[-1]["frame"]

    time_start = rows[0]["time_sec"]
    time_end = rows[-1]["time_sec"]

    duration = time_end - time_start

    # ========================================================
    # 移動量・速度
    # ========================================================

    distances = []
    pixel_speeds = []

    dx_values = []
    dy_values = []

    total_distance = 0.0
    total_dx = 0.0
    total_dy = 0.0

    max_pixel_speed = 0.0

    direction_change = 0.0

    previous_dx = None
    previous_dy = None

    for i in range(1, len(rows)):

        dx = rows[i]["center_x"] - rows[i - 1]["center_x"]

        dy = rows[i]["center_y"] - rows[i - 1]["center_y"]

        distance = math.sqrt(dx**2 + dy**2)

        dt = rows[i]["time_sec"] - rows[i - 1]["time_sec"]

        # ----------------------------------------------------
        # 移動距離
        # ----------------------------------------------------

        total_distance += distance

        distances.append(distance)

        dx_values.append(dx)
        dy_values.append(dy)

        total_dx += dx
        total_dy += dy

        # ----------------------------------------------------
        # 画面上の速度
        # ----------------------------------------------------

        if dt > 0:

            pixel_speed = distance / dt

            pixel_speeds.append(pixel_speed)

            if pixel_speed > max_pixel_speed:
                max_pixel_speed = pixel_speed

        # ----------------------------------------------------
        # 方向変化
        # ----------------------------------------------------

        if previous_dx is not None:

            cross = previous_dx * dy - previous_dy * dx

            if cross != 0:
                direction_change += 1

        previous_dx = dx
        previous_dy = dy

    # ========================================================
    # 始点→終点
    # ========================================================

    start_end_dx = rows[-1]["center_x"] - rows[0]["center_x"]

    start_end_dy = rows[-1]["center_y"] - rows[0]["center_y"]

    start_end_distance = math.sqrt(start_end_dx**2 + start_end_dy**2)

    # ========================================================
    # 画面内の移動範囲
    # ========================================================

    x_values = [row["center_x"] for row in rows]

    y_values = [row["center_y"] for row in rows]

    x_range = max(x_values) - min(x_values)

    y_range = max(y_values) - min(y_values)

    # ========================================================
    # 平均画面速度
    # ========================================================

    if duration > 0:

        average_pixel_speed = total_distance / duration

    else:

        average_pixel_speed = 0.0

    # ========================================================
    # ステップ距離
    # ========================================================

    if distances:

        mean_step_distance = float(np.mean(distances))

        std_step_distance = float(np.std(distances))

        max_step_distance = float(np.max(distances))

        median_step_distance = float(np.median(distances))

    else:

        mean_step_distance = 0.0
        std_step_distance = 0.0
        max_step_distance = 0.0
        median_step_distance = 0.0

    # ========================================================
    # 軌道の直線性
    # ========================================================

    if total_distance > 0:

        path_straightness = start_end_distance / total_distance

    else:

        path_straightness = 0.0

    # ========================================================
    # 軌道角度
    # ========================================================

    trajectory_angle_deg = math.degrees(math.atan2(start_end_dy, start_end_dx))

    # ========================================================
    # dx / dy 統計量
    # ========================================================

    if dx_values:

        mean_dx = float(np.mean(dx_values))

        mean_dy = float(np.mean(dy_values))

        std_dx = float(np.std(dx_values))

        std_dy = float(np.std(dy_values))

    else:

        mean_dx = 0.0
        mean_dy = 0.0
        std_dx = 0.0
        std_dy = 0.0

    # ========================================================
    # 画面速度の標準偏差
    # ========================================================

    if pixel_speeds:

        std_pixel_speed = float(np.std(pixel_speeds))

    else:

        std_pixel_speed = 0.0

    # ========================================================
    # confidence
    # ========================================================

    confidence_values = [row["confidence"] for row in rows]

    mean_confidence = float(np.mean(confidence_values))

    min_confidence = float(np.min(confidence_values))

    # ========================================================
    # 特徴量を返す
    # ========================================================

    return {
        "track_success": 1,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "num_frames": len(rows),
        "duration_sec": duration,
        "total_distance_px": total_distance,
        "start_end_distance_px": start_end_distance,
        "average_pixel_speed": average_pixel_speed,
        "max_pixel_speed": max_pixel_speed,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "direction_change": direction_change,
        "start_x": rows[0]["center_x"],
        "start_y": rows[0]["center_y"],
        "end_x": rows[-1]["center_x"],
        "end_y": rows[-1]["center_y"],
        "x_range": x_range,
        "y_range": y_range,
        "mean_step_distance": mean_step_distance,
        "std_step_distance": std_step_distance,
        "max_step_distance": max_step_distance,
        "median_step_distance": median_step_distance,
        "path_straightness": path_straightness,
        "trajectory_angle_deg": trajectory_angle_deg,
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "std_dx": std_dx,
        "std_dy": std_dy,
        "std_pixel_speed": std_pixel_speed,
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
    }


# ============================================================
# 開始
# ============================================================

print("=" * 60)
print("train-11 839本 特徴量データセット作成")
print("=" * 60)

print()
print("追跡CSVフォルダ:")
print(TRACKING_DIR)

print()
print("出力:")
print(OUTPUT_CSV)


# ============================================================
# フォルダ確認
# ============================================================

if not TRACKING_DIR.exists():

    raise FileNotFoundError(f"追跡CSVフォルダがありません: " f"{TRACKING_DIR}")


# ============================================================
# CSV一覧
# ============================================================

tracking_files = sorted(TRACKING_DIR.glob("*_tracking.csv"))

print()
print("追跡CSV数:", len(tracking_files))


if not tracking_files:

    raise FileNotFoundError("追跡CSVが見つかりません")


# ============================================================
# データセット作成
# ============================================================

dataset = []

success_count = 0
skip_count = 0


for index, tracking_csv in enumerate(tracking_files, start=1):

    tracking_name = tracking_csv.name

    video_stem = tracking_name.replace("_tracking.csv", "")

    video_name = video_stem + ".mp4"

    print()
    print(f"[{index}/{len(tracking_files)}] " f"{video_name}")

    # --------------------------------------------------------
    # 球速
    # --------------------------------------------------------

    speed_kmh = extract_speed_kmh(video_name)

    if speed_kmh is None:

        print("球速ラベルなし → スキップ")

        skip_count += 1

        continue

    # --------------------------------------------------------
    # 特徴量計算
    # --------------------------------------------------------

    features = calculate_features(tracking_csv)

    if features is None:

        print("追跡データ不足 → スキップ")

        skip_count += 1

        continue

    # --------------------------------------------------------
    # 1行分作成
    # --------------------------------------------------------

    row = {
        "video": video_name,
        "speed_kmh": speed_kmh,
    }

    row.update(features)

    dataset.append(row)

    success_count += 1

    # --------------------------------------------------------
    # 簡易表示
    # --------------------------------------------------------

    print("実測球速:", f"{speed_kmh:.3f} km/h")

    print("フレーム数:", features["num_frames"])

    print("std_dy:", f"{features['std_dy']:.3f}")

    print("path_straightness:", f"{features['path_straightness']:.3f}")

    print("std_pixel_speed:", f"{features['std_pixel_speed']:.3f}")

    print("std_dx:", f"{features['std_dx']:.3f}")


# ============================================================
# CSV保存
# ============================================================

fieldnames = [
    "video",
    "speed_kmh",
    "track_success",
    "frame_start",
    "frame_end",
    "num_frames",
    "duration_sec",
    "total_distance_px",
    "start_end_distance_px",
    "average_pixel_speed",
    "max_pixel_speed",
    "total_dx",
    "total_dy",
    "direction_change",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "x_range",
    "y_range",
    "mean_step_distance",
    "std_step_distance",
    "max_step_distance",
    "median_step_distance",
    "path_straightness",
    "trajectory_angle_deg",
    "mean_dx",
    "mean_dy",
    "std_dx",
    "std_dy",
    "std_pixel_speed",
    "mean_confidence",
    "min_confidence",
]


OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    for row in dataset:

        writer.writerow(row)


# ============================================================
# 統計
# ============================================================

speeds = [row["speed_kmh"] for row in dataset]


print()
print("=" * 60)
print("839本 特徴量データセット作成完了")
print("=" * 60)

print()

print("追跡CSV数:", len(tracking_files))

print("データセット作成:", success_count)

print("スキップ:", skip_count)


if speeds:

    print()

    print("球速最低:", f"{min(speeds):.3f} km/h")

    print("球速最高:", f"{max(speeds):.3f} km/h")

    print("球速平均:", f"{np.mean(speeds):.3f} km/h")

    print("球速中央値:", f"{np.median(speeds):.3f} km/h")


print()

print("特徴量数:", len(fieldnames) - 3)

print()

print("保存先:")
print(OUTPUT_CSV)

print()
print("=" * 60)
print("処理終了")
print("=" * 60)
