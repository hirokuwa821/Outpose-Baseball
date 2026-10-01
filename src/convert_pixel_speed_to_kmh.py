import csv
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_kmh")

# ------------------------------------------------------------
# 基準物の設定
# ------------------------------------------------------------
# 動画内で実際の長さが分かっている物体について、
# 「実際の長さ[m]」と「画像上の長さ[px]」を入力する。
#
# 例：
# 実際の長さ  = 1.00 m
# 画像上の長さ = 250 px
#
# → 1 px = 1 / 250 m
#
# まだ測っていない場合は None にする。
# ------------------------------------------------------------

REFERENCE_LENGTH_M = None
REFERENCE_LENGTH_PX = None


# ============================================================
# 出力フォルダ
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# px/s → km/h 変換
# ============================================================


def pixel_speed_to_kmh(pixel_speed):

    if REFERENCE_LENGTH_M is None:
        raise ValueError("REFERENCE_LENGTH_M が設定されていません。")

    if REFERENCE_LENGTH_PX is None:
        raise ValueError("REFERENCE_LENGTH_PX が設定されていません。")

    if REFERENCE_LENGTH_PX <= 0:
        raise ValueError("REFERENCE_LENGTH_PX は0より大きくしてください。")

    # px → m
    speed_m_s = pixel_speed * (REFERENCE_LENGTH_M / REFERENCE_LENGTH_PX)

    # m/s → km/h
    speed_kmh = speed_m_s * 3.6

    return speed_kmh


# ============================================================
# CSV解析
# ============================================================

csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))

if not csv_files:
    print("追跡CSVがありません")
    print("確認:", INPUT_DIR)
    exit()


print("=" * 60)
print("投球速度変換")
print("=" * 60)

print()
print("入力:", INPUT_DIR)
print("出力:", OUTPUT_DIR)

print()

if REFERENCE_LENGTH_M is None or REFERENCE_LENGTH_PX is None:

    print("=" * 60)
    print("基準物の設定が必要です")
    print("=" * 60)

    print()
    print("例えば、動画内で")
    print("実際の長さ  = 1.00 m")
    print("画像上の長さ = 250 px")
    print("なら、以下を")
    print()
    print("REFERENCE_LENGTH_M = 1.00")
    print("REFERENCE_LENGTH_PX = 250")
    print()
    print("と設定してください。")
    print()

    exit()


# ============================================================
# 各CSV
# ============================================================

summary = []


for csv_path in csv_files:

    print()
    print("=" * 60)
    print("動画:", csv_path.stem)
    print("=" * 60)

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if len(rows) < 2:

        print("追跡点が不足しています")
        continue

    # --------------------------------------------------------
    # 速度計算
    # --------------------------------------------------------

    speeds = []

    for i in range(1, len(rows)):

        try:

            frame1 = int(rows[i - 1]["frame"])
            frame2 = int(rows[i]["frame"])

            time1 = float(rows[i - 1]["time_sec"])
            time2 = float(rows[i]["time_sec"])

            x1 = float(rows[i - 1]["cx"])
            y1 = float(rows[i - 1]["cy"])

            x2 = float(rows[i]["cx"])
            y2 = float(rows[i]["cy"])

            conf1 = float(rows[i - 1]["confidence"])
            conf2 = float(rows[i]["confidence"])

        except (KeyError, ValueError):

            continue

        dt = time2 - time1

        if dt <= 0:
            continue

        # 画面上の移動距離
        dx = x2 - x1
        dy = y2 - y1

        distance_px = (dx**2 + dy**2) ** 0.5

        # px/s
        speed_px_s = distance_px / dt

        # confidence
        mean_conf = (conf1 + conf2) / 2

        # km/h
        speed_kmh = pixel_speed_to_kmh(speed_px_s)

        speeds.append(
            {
                "frame_start": frame1,
                "frame_end": frame2,
                "time_start": time1,
                "time_end": time2,
                "distance_px": distance_px,
                "speed_px_s": speed_px_s,
                "speed_kmh": speed_kmh,
                "confidence": mean_conf,
            }
        )

    if not speeds:

        print("速度を計算できませんでした")
        continue

    # --------------------------------------------------------
    # 平均・最大
    # --------------------------------------------------------

    average_px_s = sum(x["speed_px_s"] for x in speeds) / len(speeds)

    average_kmh = sum(x["speed_kmh"] for x in speeds) / len(speeds)

    max_speed = max(speeds, key=lambda x: x["speed_kmh"])

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    output_csv = OUTPUT_DIR / f"{csv_path.stem.replace('_tracking', '')}_kmh.csv"

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:

        fieldnames = [
            "frame_start",
            "frame_end",
            "time_start",
            "time_end",
            "distance_px",
            "speed_px_s",
            "speed_kmh",
            "confidence",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for row in speeds:

            writer.writerow(
                {
                    "frame_start": row["frame_start"],
                    "frame_end": row["frame_end"],
                    "time_start": row["time_start"],
                    "time_end": row["time_end"],
                    "distance_px": f"{row['distance_px']:.3f}",
                    "speed_px_s": f"{row['speed_px_s']:.3f}",
                    "speed_kmh": f"{row['speed_kmh']:.3f}",
                    "confidence": f"{row['confidence']:.4f}",
                }
            )

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print(f"平均速度      : " f"{average_kmh:.2f} km/h")

    print(f"最大瞬間速度  : " f"{max_speed['speed_kmh']:.2f} km/h")

    print(
        f"最大速度区間  : "
        f"Frame {max_speed['frame_start']} "
        f"→ {max_speed['frame_end']}"
    )

    print(f"画面速度      : " f"{max_speed['speed_px_s']:.2f} px/s")

    print(f"confidence     : " f"{max_speed['confidence']:.3f}")

    print()
    print("CSV:", output_csv)

    summary.append(
        {
            "video": csv_path.stem,
            "average_kmh": average_kmh,
            "max_kmh": max_speed["speed_kmh"],
        }
    )


# ============================================================
# 結果まとめ
# ============================================================

print()
print("=" * 60)
print("全動画の速度変換完了")
print("=" * 60)

print()

for result in summary:

    print(
        f"{result['video']} | "
        f"平均: {result['average_kmh']:.2f} km/h | "
        f"最大: {result['max_kmh']:.2f} km/h"
    )

print()
print("出力:", OUTPUT_DIR)
