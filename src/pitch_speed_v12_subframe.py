"""
投球速度解析 v12
サブフレーム時間推定版

目的:
手動で指定したリリース/ホーム到達フレームを基準に、
ボール軌跡からフレーム間の到達時刻を線形補間して、
サブフレーム精度で飛行時間を推定する。

投球距離:
18.44 m - 1.95 m = 16.49 m

注意:
このバージョンでは、ホーム到達フレーム自体は手動設定する。
そのフレームの前後のボール位置から、フレーム間の到達時刻を補間する。
"""

import csv
from pathlib import Path

import numpy as np

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v12")

# 投球距離
PITCH_DISTANCE_M = 16.49

# 使用する最低confidence
MIN_CONFIDENCE = 0.3

# 現在の基準フレーム
# (release_frame, home_frame)
BASE_FRAMES = {
    "pitch_00015_80": (429, 455),
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 94),
    "pitch_98_2": (66, 88),
}


# ============================================================
# CSV読み込み
# ============================================================


def load_tracking_csv(csv_path):

    rows = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                frame = int(row["frame"])
                time_sec = float(row["time_sec"])
                cx = float(row["cx"])
                cy = float(row["cy"])
                confidence = float(row["confidence"])
                interpolated = int(row.get("interpolated", 0))

            except (ValueError, KeyError):
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


# ============================================================
# フレーム間のサブフレーム時刻推定
# ============================================================


def interpolate_time(
    frame_a,
    time_a,
    distance_a,
    frame_b,
    time_b,
    distance_b,
    target_distance,
):

    if distance_b == distance_a:
        return None

    ratio = (target_distance - distance_a) / (distance_b - distance_a)

    ratio = max(0.0, min(1.0, ratio))

    return time_a + ratio * (time_b - time_a)


# ============================================================
# 軌跡距離
# ============================================================


def calculate_cumulative_distance(rows):

    distance = 0.0

    cumulative = []

    previous = None

    for row in rows:

        if previous is None:

            cumulative.append(0.0)

            previous = row
            continue

        dx = row["cx"] - previous["cx"]

        dy = row["cy"] - previous["cy"]

        step = np.sqrt(dx * dx + dy * dy)

        distance += step

        cumulative.append(distance)

        previous = row

    return np.array(cumulative)


# ============================================================
# 指定フレーム付近の時刻推定
# ============================================================


def estimate_subframe_time(
    rows,
    cumulative_distance,
    target_frame,
):

    # 完全一致するフレームを探す
    exact = [i for i, row in enumerate(rows) if row["frame"] == target_frame]

    if exact:

        i = exact[0]

        return (rows[i]["time_sec"], float(i), "exact")

    # 前後の2点を探す
    before = None
    after = None

    for i, row in enumerate(rows):

        if row["frame"] < target_frame:

            before = i

        elif row["frame"] > target_frame:

            after = i
            break

    if before is None or after is None:

        return None

    frame_a = rows[before]["frame"]
    frame_b = rows[after]["frame"]

    time_a = rows[before]["time_sec"]
    time_b = rows[after]["time_sec"]

    # フレーム位置による線形補間
    ratio = (target_frame - frame_a) / (frame_b - frame_a)

    time_est = time_a + ratio * (time_b - time_a)

    return (time_est, before + ratio * (after - before), "interpolated")


# ============================================================
# CSV保存
# ============================================================


def save_result(
    output_path,
    result,
):

    fieldnames = [
        "video",
        "release_frame",
        "home_frame",
        "release_time_sec",
        "home_time_sec",
        "flight_time_sec",
        "pitch_distance_m",
        "speed_mps",
        "speed_kmh",
        "release_method",
        "home_method",
        "valid_points",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(result)


# ============================================================
# メイン
# ============================================================


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("投球速度解析 v12")
    print("サブフレーム時間推定")
    print("=" * 60)

    print()
    print(f"入力: {INPUT_DIR}")

    print(f"出力: {OUTPUT_DIR}")

    print(f"投球距離: " f"{PITCH_DISTANCE_M:.3f} m")

    print(f"最低confidence: " f"{MIN_CONFIDENCE}")

    print("=" * 60)

    summary = []

    for video_name, (release_frame, home_frame) in BASE_FRAMES.items():

        print()
        print("=" * 60)
        print(f"動画: {video_name}")
        print("=" * 60)

        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"

        if not csv_path.exists():

            print(f"CSVが見つかりません:" f" {csv_path}")

            continue

        rows = load_tracking_csv(csv_path)

        if len(rows) < 3:

            print("追跡点が不足しています")

            continue

        # confidenceでフィルタ
        valid_rows = [r for r in rows if r["confidence"] >= MIN_CONFIDENCE]

        if len(valid_rows) < 3:

            print("有効追跡点が不足しています")

            continue

        cumulative = calculate_cumulative_distance(valid_rows)

        # ----------------------------------------------------
        # リリース時刻
        # ----------------------------------------------------

        release_result = estimate_subframe_time(valid_rows, cumulative, release_frame)

        # ----------------------------------------------------
        # ホーム到達時刻
        # ----------------------------------------------------

        home_result = estimate_subframe_time(valid_rows, cumulative, home_frame)

        if release_result is None or home_result is None:

            print("サブフレーム時刻を" "推定できません")

            continue

        release_time = release_result[0]

        home_time = home_result[0]

        release_method = release_result[2]

        home_method = home_result[2]

        flight_time = home_time - release_time

        if flight_time <= 0:

            print("飛行時間が不正です")

            continue

        # ----------------------------------------------------
        # 速度
        # ----------------------------------------------------

        speed_mps = PITCH_DISTANCE_M / flight_time

        speed_kmh = speed_mps * 3.6

        # ----------------------------------------------------
        # 表示
        # ----------------------------------------------------

        print()
        print("追跡点数:" f" {len(valid_rows)}")

        print()
        print("------------------------------------------")

        print("時間解析")

        print("------------------------------------------")

        print(f"Release Frame : " f"{release_frame}")

        print(f"Release Time  : " f"{release_time:.6f} sec")

        print(f"Release method: " f"{release_method}")

        print()

        print(f"Home Frame    : " f"{home_frame}")

        print(f"Home Time     : " f"{home_time:.6f} sec")

        print(f"Home method   : " f"{home_method}")

        print()

        print(f"飛行時間      : " f"{flight_time:.6f} sec")

        print()
        print("------------------------------------------")

        print("速度解析")

        print("------------------------------------------")

        print(f"投球距離      : " f"{PITCH_DISTANCE_M:.3f} m")

        print(f"速度          : " f"{speed_mps:.3f} m/s")

        print(f"速度          : " f"{speed_kmh:.2f} km/h")

        print("------------------------------------------")

        # ----------------------------------------------------
        # 個別CSV
        # ----------------------------------------------------

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "home_frame": home_frame,
            "release_time_sec": release_time,
            "home_time_sec": home_time,
            "flight_time_sec": flight_time,
            "pitch_distance_m": PITCH_DISTANCE_M,
            "speed_mps": speed_mps,
            "speed_kmh": speed_kmh,
            "release_method": release_method,
            "home_method": home_method,
            "valid_points": len(valid_rows),
        }

        output_csv = OUTPUT_DIR / f"{video_name}_v12.csv"

        save_result(output_csv, result)

        print()
        print(f"CSV: {output_csv}")

        summary.append(result)

    # ========================================================
    # まとめCSV
    # ========================================================

    summary_csv = OUTPUT_DIR / "pitch_speed_v12_summary.csv"

    fieldnames = [
        "video",
        "release_frame",
        "home_frame",
        "release_time_sec",
        "home_time_sec",
        "flight_time_sec",
        "pitch_distance_m",
        "speed_mps",
        "speed_kmh",
        "release_method",
        "home_method",
        "valid_points",
    ]

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 60)
    print("v12 全動画解析完了")
    print("=" * 60)

    for r in summary:

        print(
            f"{r['video']} | "
            f"{r['speed_kmh']:.2f} km/h | "
            f"飛行時間 "
            f"{r['flight_time_sec']:.6f} sec"
        )

    print()
    print(f"まとめCSV: {summary_csv}")

    print(f"出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
