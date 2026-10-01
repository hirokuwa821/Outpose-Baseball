import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# 投球速度解析 v3
#
# ・v5の追跡CSVを読み込む
# ・confidenceが低い点を除外
# ・補間点を除外して実検出だけでも計算可能
# ・フレーム間速度を計算
# ・平均速度 / 最大速度を計算
# ・px/s → km/h 変換にも対応
# ・速度グラフを保存
# ============================================================


# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")

OUTPUT_DIR = Path("output/pitch_speed_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# confidenceの最低値
# ------------------------------------------------------------

MIN_CONFIDENCE = 0.30


# ------------------------------------------------------------
# 補間点を速度計算から除外する
#
# True:
#   実検出点だけを使用
#
# False:
#   補間点も使用
# ------------------------------------------------------------

USE_INTERPOLATED = False


# ------------------------------------------------------------
# 実距離変換
#
# まだ設定しない場合は None
#
# 例えば
#
#     100 px = 1 m
#
# なら
#
#     PIXELS_PER_METER = 100
#
# とする
# ------------------------------------------------------------

PIXELS_PER_METER = None


# ============================================================
# 関数
# ============================================================


def read_tracking_csv(csv_path):

    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                frame = int(row["frame"])

                time_sec = float(row["time_sec"])

                cx = float(row["cx"])

                cy = float(row["cy"])

                confidence = float(row["confidence"])

                interpolated = int(row.get("interpolated", 0))

            except Exception:

                continue

            rows.append(
                {
                    "frame": frame,
                    "time_sec": time_sec,
                    "cx": cx,
                    "cy": cy,
                    "confidence": confidence,
                    "interpolated": interpolated,
                }
            )

    return rows


def calculate_speed(p1, p2):

    dx = p2["cx"] - p1["cx"]

    dy = p2["cy"] - p1["cy"]

    distance_px = math.sqrt(dx**2 + dy**2)

    dt = p2["time_sec"] - p1["time_sec"]

    if dt <= 0:

        return None

    speed_px = distance_px / dt

    return (speed_px, distance_px, dt)


def px_to_kmh(px_per_sec):

    if PIXELS_PER_METER is None:

        return None

    meter_per_sec = px_per_sec / PIXELS_PER_METER

    km_per_hour = meter_per_sec * 3.6

    return km_per_hour


# ============================================================
# メイン
# ============================================================


print("=" * 60)
print("投球速度解析 v3")
print("=" * 60)

print()
print("入力:", INPUT_DIR)
print("出力:", OUTPUT_DIR)
print("最低confidence:", MIN_CONFIDENCE)

if PIXELS_PER_METER is None:

    print("実距離変換: 未設定 " "(px/sのみ)")

else:

    print("実距離変換:", PIXELS_PER_METER, "px/m")


csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))


if not csv_files:

    print()
    print("追跡CSVがありません")

    print("確認してください:", INPUT_DIR)

    raise SystemExit


all_results = []


for csv_path in csv_files:

    name = csv_path.stem.replace("_tracking", "")

    print()
    print("=" * 60)
    print("動画:", name)
    print("=" * 60)

    # --------------------------------------------------------
    # CSV読み込み
    # --------------------------------------------------------

    rows = read_tracking_csv(csv_path)

    if len(rows) < 2:

        print("追跡点が不足しています")

        continue

    print("追跡点数:", len(rows))

    # --------------------------------------------------------
    # 速度計算
    # --------------------------------------------------------

    speed_rows = []

    for i in range(len(rows) - 1):

        p1 = rows[i]

        p2 = rows[i + 1]

        # confidence
        if p1["confidence"] < MIN_CONFIDENCE:

            continue

        if p2["confidence"] < MIN_CONFIDENCE:

            continue

        # 補間点を除外
        if not USE_INTERPOLATED:

            if p1["interpolated"] == 1 or p2["interpolated"] == 1:

                continue

        result = calculate_speed(p1, p2)

        if result is None:

            continue

        speed_px, distance_px, dt = result

        speed_kmh = px_to_kmh(speed_px)

        speed_rows.append(
            {
                "frame_start": p1["frame"],
                "frame_end": p2["frame"],
                "time_start": p1["time_sec"],
                "time_end": p2["time_sec"],
                "distance_px": distance_px,
                "dt_sec": dt,
                "px_per_sec": speed_px,
                "kmh": "" if speed_kmh is None else speed_kmh,
                "confidence_start": p1["confidence"],
                "confidence_end": p2["confidence"],
                "interpolated_start": p1["interpolated"],
                "interpolated_end": p2["interpolated"],
            }
        )

    if not speed_rows:

        print("有効な速度データがありません")

        continue

    # --------------------------------------------------------
    # 平均速度
    # --------------------------------------------------------

    speeds = [x["px_per_sec"] for x in speed_rows]

    mean_speed = sum(speeds) / len(speeds)

    # --------------------------------------------------------
    # 最大速度
    # --------------------------------------------------------

    max_row = max(speed_rows, key=lambda x: x["px_per_sec"])

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print("有効速度区間:", len(speed_rows))

    print(f"平均速度      : " f"{mean_speed:.2f} px/s")

    print(f"最大瞬間速度  : " f"{max_row['px_per_sec']:.2f} px/s")

    print(
        "最大速度区間  : "
        f"Frame "
        f"{max_row['frame_start']}"
        f" → "
        f"{max_row['frame_end']}"
    )

    print(f"距離          : " f"{max_row['distance_px']:.2f} px")

    print(f"時間          : " f"{max_row['dt_sec']:.5f} sec")

    print(f"confidence    : " f"{max_row['confidence_start']:.3f}")

    # --------------------------------------------------------
    # km/h
    # --------------------------------------------------------

    if PIXELS_PER_METER is not None:

        kmh_values = [x["kmh"] for x in speed_rows if x["kmh"] != ""]

        mean_kmh = sum(kmh_values) / len(kmh_values)

        max_kmh = max(kmh_values)

        print()
        print("------------------------------------------")
        print("実距離換算")
        print("------------------------------------------")

        print(f"平均速度      : " f"{mean_kmh:.2f} km/h")

        print(f"最大瞬間速度  : " f"{max_kmh:.2f} km/h")

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    output_csv = OUTPUT_DIR / f"{name}_speed_v3.csv"

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:

        fieldnames = [
            "frame_start",
            "frame_end",
            "time_start",
            "time_end",
            "distance_px",
            "dt_sec",
            "px_per_sec",
            "kmh",
            "confidence_start",
            "confidence_end",
            "interpolated_start",
            "interpolated_end",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in speed_rows:

            writer.writerow(row)

    print()
    print("CSV:", output_csv)

    # --------------------------------------------------------
    # グラフ
    # --------------------------------------------------------

    times = [x["time_start"] for x in speed_rows]

    if PIXELS_PER_METER is None:

        values = [x["px_per_sec"] for x in speed_rows]

        ylabel = "Speed (px/s)"

    else:

        values = [x["kmh"] for x in speed_rows]

        ylabel = "Speed (km/h)"

    plt.figure(figsize=(10, 5))

    plt.plot(times, values, marker="o", markersize=3)

    plt.xlabel("Time (s)")

    plt.ylabel(ylabel)

    plt.title(f"{name} - Ball Speed")

    plt.grid(True)

    plt.tight_layout()

    graph_path = OUTPUT_DIR / f"{name}_speed_v3.png"

    plt.savefig(graph_path, dpi=150)

    plt.close()

    print("グラフ:", graph_path)

    # --------------------------------------------------------
    # 結果保存
    # --------------------------------------------------------

    all_results.append(
        {
            "name": name,
            "tracking_points": len(rows),
            "valid_intervals": len(speed_rows),
            "mean_px_per_sec": mean_speed,
            "max_px_per_sec": max_row["px_per_sec"],
        }
    )


# ============================================================
# 全動画まとめ
# ============================================================


print()
print("=" * 60)
print("全動画の速度解析完了")
print("=" * 60)


for result in all_results:

    print()

    print(
        result["name"],
        "|",
        "平均:",
        f"{result['mean_px_per_sec']:.2f}",
        "px/s",
        "|",
        "最大:",
        f"{result['max_px_per_sec']:.2f}",
        "px/s",
    )
