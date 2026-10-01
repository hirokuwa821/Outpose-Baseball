import csv
import math
from pathlib import Path

# ============================================================
# 投球速度解析 v2
# v5追跡CSV対応
#
# CSV:
# frame,time_sec,cx,cy,confidence,interpolated
# ============================================================

TRACK_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    "pitch_102_1_tracking.csv",
    "pitch_98_1_tracking.csv",
    "pitch_98_2_tracking.csv",
]

# ------------------------------------------------------------
# 実空間への変換
# ------------------------------------------------------------
# 現在は動画上のピクセル速度を計算する。
#
# 実際の km/h を求めるには、
# 「動画上の何pxが実際の何mに相当するか」
# というスケール情報が必要。
#
# そのため、まず pixel/s を計算する。
# ------------------------------------------------------------


def load_csv(csv_path):

    data = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        required = [
            "frame",
            "time_sec",
            "cx",
            "cy",
            "confidence",
            "interpolated",
        ]

        for col in required:

            if col not in reader.fieldnames:
                raise ValueError(
                    f"必要な列がありません: {col}\n" f"CSV列: {reader.fieldnames}"
                )

        for row in reader:

            data.append(
                {
                    "frame": int(row["frame"]),
                    "time_sec": float(row["time_sec"]),
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "confidence": float(row["confidence"]),
                    "interpolated": int(row["interpolated"]),
                }
            )

    return data


def calculate_velocity(data):

    results = []

    for i in range(1, len(data)):

        p1 = data[i - 1]
        p2 = data[i]

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance_px = math.sqrt(dx * dx + dy * dy)

        dt = p2["time_sec"] - p1["time_sec"]

        if dt <= 0:
            continue

        velocity_px_s = distance_px / dt

        results.append(
            {
                "frame_start": p1["frame"],
                "frame_end": p2["frame"],
                "time_start": p1["time_sec"],
                "time_end": p2["time_sec"],
                "distance_px": distance_px,
                "dt": dt,
                "velocity_px_s": velocity_px_s,
                "confidence": min(p1["confidence"], p2["confidence"]),
                "interpolated": (p1["interpolated"] or p2["interpolated"]),
            }
        )

    return results


def analyze(csv_path):

    print()
    print("=" * 60)
    print("動画:", csv_path.stem.replace("_tracking", ""))
    print("=" * 60)

    data = load_csv(csv_path)

    print("追跡点数:", len(data))

    if len(data) < 2:

        print("追跡点が不足しています")

        return None

    velocities = calculate_velocity(data)

    if not velocities:

        print("速度を計算できませんでした")

        return None

    # --------------------------------------------------------
    # 最大速度
    # --------------------------------------------------------

    max_result = max(velocities, key=lambda x: x["velocity_px_s"])

    # --------------------------------------------------------
    # 平均速度
    # --------------------------------------------------------

    avg_velocity = sum(x["velocity_px_s"] for x in velocities) / len(velocities)

    # --------------------------------------------------------
    # 実検出点だけで計算した速度
    # --------------------------------------------------------

    real_velocities = [x["velocity_px_s"] for x in velocities if not x["interpolated"]]

    if real_velocities:

        avg_real_velocity = sum(real_velocities) / len(real_velocities)

    else:

        avg_real_velocity = None

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print(f"平均速度      : " f"{avg_velocity:.2f} px/s")

    print(f"最大瞬間速度  : " f"{max_result['velocity_px_s']:.2f} px/s")

    print(
        f"最大速度区間  : "
        f"Frame {max_result['frame_start']} "
        f"→ {max_result['frame_end']}"
    )

    print(f"距離          : " f"{max_result['distance_px']:.2f} px")

    print(f"時間          : " f"{max_result['dt']:.5f} sec")

    print(f"confidence     : " f"{max_result['confidence']:.3f}")

    if avg_real_velocity is not None:

        print(f"実検出のみ平均: " f"{avg_real_velocity:.2f} px/s")

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / f"{csv_path.stem}_speed.csv"

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame_start",
                "frame_end",
                "time_start",
                "time_end",
                "distance_px",
                "dt_sec",
                "velocity_px_s",
                "confidence",
                "interpolated",
            ]
        )

        for r in velocities:

            writer.writerow(
                [
                    r["frame_start"],
                    r["frame_end"],
                    f"{r['time_start']:.6f}",
                    f"{r['time_end']:.6f}",
                    f"{r['distance_px']:.4f}",
                    f"{r['dt']:.6f}",
                    f"{r['velocity_px_s']:.4f}",
                    f"{r['confidence']:.4f}",
                    r["interpolated"],
                ]
            )

    print()
    print("保存:", output_path)

    return {
        "average": avg_velocity,
        "maximum": max_result["velocity_px_s"],
    }


# ============================================================
# メイン
# ============================================================

print()
print("=" * 60)
print("投球速度解析 v2")
print("=" * 60)

all_results = {}

for filename in CSV_FILES:

    csv_path = TRACK_DIR / filename

    if not csv_path.exists():

        print()
        print("CSVがありません:", csv_path)
        continue

    result = analyze(csv_path)

    if result is not None:

        all_results[filename] = result


print()
print("=" * 60)
print("全動画の速度解析完了")
print("=" * 60)

print()

for filename, result in all_results.items():

    print(
        filename,
        f"| 平均: {result['average']:.2f} px/s",
        f"| 最大: {result['maximum']:.2f} px/s",
    )
