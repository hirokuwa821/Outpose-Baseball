import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 実際の身長
HEIGHT_CM = 175.0

# 各動画での「投手の身長」が画像上で何pxに見えるか
# ここを実測して入力する
PLAYER_HEIGHT_PX = {
    "pitch_102_1": 350.0,
    "pitch_98_1": 350.0,
    "pitch_98_2": 350.0,
}

# 速度計算に使う最低confidence
MIN_CONFIDENCE = 0.3


# ============================================================
# CSV読み込み
# ============================================================


def load_tracking(csv_path):

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(
                {
                    "frame": int(row["frame"]),
                    "time_sec": float(row["time_sec"]),
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "confidence": float(row["confidence"]),
                    "interpolated": int(row["interpolated"]),
                }
            )

    return rows


# ============================================================
# 速度解析
# ============================================================

for csv_path in sorted(INPUT_DIR.glob("*_tracking.csv")):

    name = csv_path.stem.replace("_tracking", "")

    print()
    print("=" * 60)
    print(f"動画: {name}")
    print("=" * 60)

    if name not in PLAYER_HEIGHT_PX:

        print("投手身長pxが設定されていません")
        continue

    player_height_px = PLAYER_HEIGHT_PX[name]

    if player_height_px <= 0:

        print("投手身長pxが不正です")
        continue

    rows = load_tracking(csv_path)

    print(f"追跡点数: {len(rows)}")
    print(f"投手身長: {HEIGHT_CM} cm")
    print(f"画像上の身長: {player_height_px} px")

    # --------------------------------------------------------
    # px → 実距離
    # --------------------------------------------------------

    cm_per_px = HEIGHT_CM / player_height_px
    m_per_px = cm_per_px / 100.0

    print(f"換算係数: {cm_per_px:.6f} cm/px")

    speeds = []

    for i in range(len(rows) - 1):

        p1 = rows[i]
        p2 = rows[i + 1]

        # confidenceが低い点を除外
        if p1["confidence"] < MIN_CONFIDENCE:
            continue

        if p2["confidence"] < MIN_CONFIDENCE:
            continue

        dt = p2["time_sec"] - p1["time_sec"]

        if dt <= 0:
            continue

        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]

        distance_px = (dx**2 + dy**2) ** 0.5

        # px/s
        speed_px_s = distance_px / dt

        # m/s
        speed_m_s = speed_px_s * m_per_px

        # km/h
        speed_kmh = speed_m_s * 3.6

        speeds.append(
            {
                "frame1": p1["frame"],
                "frame2": p2["frame"],
                "time1": p1["time_sec"],
                "time2": p2["time_sec"],
                "distance_px": distance_px,
                "speed_px_s": speed_px_s,
                "speed_m_s": speed_m_s,
                "speed_kmh": speed_kmh,
                "confidence": min(p1["confidence"], p2["confidence"]),
                "interpolated": (p1["interpolated"] or p2["interpolated"]),
            }
        )

    if not speeds:

        print("有効な速度区間がありません")
        continue

    # ========================================================
    # 結果
    # ========================================================

    average_kmh = sum(x["speed_kmh"] for x in speeds) / len(speeds)

    max_item = max(speeds, key=lambda x: x["speed_kmh"])

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print(f"平均速度      : " f"{average_kmh:.2f} km/h")

    print(f"最大瞬間速度  : " f"{max_item['speed_kmh']:.2f} km/h")

    print(f"最大速度区間  : " f"Frame {max_item['frame1']} " f"→ {max_item['frame2']}")

    print(f"距離          : " f"{max_item['distance_px']:.2f} px")

    print(f"速度          : " f"{max_item['speed_px_s']:.2f} px/s")

    print(f"confidence     : " f"{max_item['confidence']:.3f}")

    # ========================================================
    # CSV保存
    # ========================================================

    output_csv = OUTPUT_DIR / f"{name}_speed_v4.csv"

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame1",
                "frame2",
                "time1",
                "time2",
                "distance_px",
                "speed_px_s",
                "speed_m_s",
                "speed_kmh",
                "confidence",
                "interpolated",
            ]
        )

        for x in speeds:

            writer.writerow(
                [
                    x["frame1"],
                    x["frame2"],
                    f"{x['time1']:.6f}",
                    f"{x['time2']:.6f}",
                    f"{x['distance_px']:.3f}",
                    f"{x['speed_px_s']:.3f}",
                    f"{x['speed_m_s']:.3f}",
                    f"{x['speed_kmh']:.3f}",
                    f"{x['confidence']:.3f}",
                    x["interpolated"],
                ]
            )

    # ========================================================
    # グラフ
    # ========================================================

    frames = [x["frame2"] for x in speeds]

    velocity = [x["speed_kmh"] for x in speeds]

    plt.figure(figsize=(10, 5))

    plt.plot(frames, velocity, marker="o")

    plt.xlabel("Frame")
    plt.ylabel("Velocity (km/h)")

    plt.title(f"{name} Ball Velocity")

    plt.grid(True)

    plt.tight_layout()

    output_png = OUTPUT_DIR / f"{name}_speed_v4.png"

    plt.savefig(output_png, dpi=150)

    plt.close()

    print()
    print(f"CSV: {output_csv}")
    print(f"グラフ: {output_png}")


print()
print("=" * 60)
print("全動画の速度解析完了")
print("=" * 60)
