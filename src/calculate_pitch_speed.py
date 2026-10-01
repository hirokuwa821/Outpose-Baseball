import csv
import math
from pathlib import Path

# ============================================================
# 設定
# ============================================================

TRACKING_DIR = Path("output/ball_tracking_v5")

VIDEOS = [
    "pitch_102_1",
    "pitch_98_1",
    "pitch_98_2",
]

# 実際の動画FPS
FPS = {
    "pitch_102_1": 29.954954954954953,
    "pitch_98_1": 29.938587512794268,
    "pitch_98_2": 29.991622451829098,
}


# ============================================================
# CSV読み込み
# ============================================================


def load_tracking_csv(path):

    points = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                frame = int(row["frame"])
                x = float(row["x"])
                y = float(row["y"])

                confidence = float(row.get("confidence", 0))

                interpolated = row.get("interpolated", "0")

                points.append(
                    {
                        "frame": frame,
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                        "interpolated": interpolated,
                    }
                )

            except (ValueError, KeyError):
                continue

    return points


# ============================================================
# フレーム間の移動量
# ============================================================


def calculate_motion(points, fps):

    results = []

    for i in range(1, len(points)):

        p1 = points[i - 1]
        p2 = points[i]

        frame_diff = p2["frame"] - p1["frame"]

        if frame_diff <= 0:
            continue

        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]

        pixel_distance = math.sqrt(dx**2 + dy**2)

        time_sec = frame_diff / fps

        pixel_speed = pixel_distance / time_sec

        results.append(
            {
                "frame1": p1["frame"],
                "frame2": p2["frame"],
                "dx": dx,
                "dy": dy,
                "pixel_distance": pixel_distance,
                "time_sec": time_sec,
                "pixel_speed": pixel_speed,
            }
        )

    return results


# ============================================================
# メイン
# ============================================================

print()
print("=" * 60)
print("投球速度解析")
print("=" * 60)


for video_name in VIDEOS:

    csv_path = TRACKING_DIR / f"{video_name}_tracking.csv"

    print()
    print("=" * 60)
    print(f"動画: {video_name}")
    print("=" * 60)

    if not csv_path.exists():

        print("CSVがありません:")
        print(csv_path)

        continue

    fps = FPS[video_name]

    points = load_tracking_csv(csv_path)

    if len(points) < 2:

        print("追跡点が不足しています")

        continue

    print(f"追跡点数: {len(points)}")
    print(f"Frame: {points[0]['frame']} → " f"{points[-1]['frame']}")

    motions = calculate_motion(points, fps)

    if not motions:

        print("移動量を計算できません")

        continue

    # --------------------------------------------------------
    # 統計
    # --------------------------------------------------------

    total_distance = sum(m["pixel_distance"] for m in motions)

    total_time = sum(m["time_sec"] for m in motions)

    average_pixel_speed = total_distance / total_time

    max_pixel_speed = max(m["pixel_speed"] for m in motions)

    # --------------------------------------------------------
    # 結果
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("ピクセル速度")
    print("------------------------------------------")

    print(f"総移動距離       : " f"{total_distance:.2f} px")

    print(f"追跡時間         : " f"{total_time:.4f} sec")

    print(f"平均ピクセル速度 : " f"{average_pixel_speed:.2f} px/s")

    print(f"最大ピクセル速度 : " f"{max_pixel_speed:.2f} px/s")

    # --------------------------------------------------------
    # 各フレームの詳細
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("フレーム間速度")
    print("------------------------------------------")

    for m in motions:

        print(
            f"{m['frame1']:3d} → "
            f"{m['frame2']:3d} | "
            f"移動 {m['pixel_distance']:7.2f} px | "
            f"速度 {m['pixel_speed']:8.2f} px/s"
        )

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    output_path = TRACKING_DIR / f"{video_name}_speed_analysis.csv"

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame1",
                "frame2",
                "dx",
                "dy",
                "pixel_distance",
                "time_sec",
                "pixel_speed",
            ]
        )

        for m in motions:

            writer.writerow(
                [
                    m["frame1"],
                    m["frame2"],
                    f"{m['dx']:.4f}",
                    f"{m['dy']:.4f}",
                    f"{m['pixel_distance']:.4f}",
                    f"{m['time_sec']:.6f}",
                    f"{m['pixel_speed']:.4f}",
                ]
            )

    print()
    print("解析CSV:")
    print(output_path)


print()
print("=" * 60)
print("全動画の速度解析完了")
print("=" * 60)
