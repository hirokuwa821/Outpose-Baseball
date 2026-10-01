"""
pitch_speed_v12_1_subframe.py

投球速度解析 v12.1
サブフレーム時間推定・改良版

変更点:
- 指定Release/HomeフレームがCSVに存在しなくても解析可能
- 指定フレームの前後にある実追跡点から時刻を補間
- CSVの実際の追跡範囲を自動確認
- exact / interpolated を表示
- v11と同じ投球距離 16.49 m を使用
"""

import csv
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v12_1")

PITCH_DISTANCE_M = 16.49
MIN_CONFIDENCE = 0.3

# 手動で指定した基準フレーム
BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 94),
    "pitch_98_2": (66, 88),
}


# ============================================================
# CSV読み込み
# ============================================================


def load_csv(path):

    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:
                frame = int(row["frame"])
                time_sec = float(row["time_sec"])
                cx = float(row["cx"])
                cy = float(row["cy"])
                confidence = float(row["confidence"])
            except (ValueError, KeyError):
                continue

            if confidence < MIN_CONFIDENCE:
                continue

            rows.append(
                {
                    "frame": frame,
                    "time_sec": time_sec,
                    "cx": cx,
                    "cy": cy,
                    "confidence": confidence,
                }
            )

    rows.sort(key=lambda x: x["frame"])

    return rows


# ============================================================
# 指定フレームの時刻を推定
# ============================================================


def estimate_time(rows, target_frame):

    if not rows:
        return None

    # --------------------------------------------------------
    # 完全一致
    # --------------------------------------------------------

    for row in rows:

        if row["frame"] == target_frame:

            return {
                "time_sec": row["time_sec"],
                "frame_float": float(target_frame),
                "method": "exact",
                "before_frame": target_frame,
                "after_frame": target_frame,
            }

    # --------------------------------------------------------
    # 前後の追跡点を探す
    # --------------------------------------------------------

    before = None
    after = None

    for row in rows:

        if row["frame"] < target_frame:
            before = row

        elif row["frame"] > target_frame:
            after = row
            break

    # --------------------------------------------------------
    # 前後両方がある場合
    # --------------------------------------------------------

    if before is not None and after is not None:

        frame1 = before["frame"]
        frame2 = after["frame"]

        time1 = before["time_sec"]
        time2 = after["time_sec"]

        frame_ratio = (target_frame - frame1) / (frame2 - frame1)

        time_est = time1 + frame_ratio * (time2 - time1)

        return {
            "time_sec": time_est,
            "frame_float": float(target_frame),
            "method": "interpolated",
            "before_frame": frame1,
            "after_frame": frame2,
        }

    # --------------------------------------------------------
    # 指定フレームが追跡範囲より前の場合
    # --------------------------------------------------------

    if before is None:

        first = rows[0]

        return {
            "time_sec": first["time_sec"],
            "frame_float": float(first["frame"]),
            "method": "clamped_before",
            "before_frame": None,
            "after_frame": first["frame"],
        }

    # --------------------------------------------------------
    # 指定フレームが追跡範囲より後の場合
    # --------------------------------------------------------

    if after is None:

        last = rows[-1]

        return {
            "time_sec": last["time_sec"],
            "frame_float": float(last["frame"]),
            "method": "clamped_after",
            "before_frame": last["frame"],
            "after_frame": None,
        }

    return None


# ============================================================
# 個別結果保存
# ============================================================


def save_result(path, result):

    fields = [
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
        "release_before_frame",
        "release_after_frame",
        "home_before_frame",
        "home_after_frame",
        "valid_points",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()
        writer.writerow(result)


# ============================================================
# メイン
# ============================================================


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("投球速度解析 v12.1")
    print("サブフレーム時間推定・改良版")
    print("=" * 60)

    print()
    print(f"入力: {INPUT_DIR}")

    print(f"出力: {OUTPUT_DIR}")

    print(f"投球距離: {PITCH_DISTANCE_M:.3f} m")

    print(f"最低confidence: {MIN_CONFIDENCE}")

    print("=" * 60)

    summary = []

    for video_name, (release_frame, home_frame) in BASE_FRAMES.items():

        print()
        print("=" * 60)
        print(f"動画: {video_name}")
        print("=" * 60)

        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"

        print(f"CSV: {csv_path}")

        if not csv_path.exists():

            print()
            print("CSVが見つかりません")

            print("この動画はスキップします")

            continue

        rows = load_csv(csv_path)

        print(f"有効追跡点数: {len(rows)}")

        if len(rows) < 2:

            print("追跡点が不足しています")

            continue

        print(f"追跡範囲: " f"Frame {rows[0]['frame']}" f" → " f"{rows[-1]['frame']}")

        # ----------------------------------------------------
        # Release
        # ----------------------------------------------------

        release = estimate_time(rows, release_frame)

        # ----------------------------------------------------
        # Home
        # ----------------------------------------------------

        home = estimate_time(rows, home_frame)

        if release is None or home is None:

            print("時刻推定に失敗しました")

            continue

        # ----------------------------------------------------
        # 飛行時間
        # ----------------------------------------------------

        flight_time = home["time_sec"] - release["time_sec"]

        if flight_time <= 0:

            print("飛行時間が0以下です")

            continue

        # ----------------------------------------------------
        # 速度
        # ----------------------------------------------------

        speed_mps = PITCH_DISTANCE_M / flight_time

        speed_kmh = speed_mps * 3.6

        # ----------------------------------------------------
        # 結果表示
        # ----------------------------------------------------

        print()
        print("------------------------------------------")

        print("Release")

        print(f"指定Frame    : " f"{release_frame}")

        print(f"推定時刻     : " f"{release['time_sec']:.6f} sec")

        print(f"方法         : " f"{release['method']}")

        if release["before_frame"] is not None:

            print(f"前Frame      : " f"{release['before_frame']}")

        if release["after_frame"] is not None:

            print(f"後Frame      : " f"{release['after_frame']}")

        print()
        print("Home")

        print(f"指定Frame    : " f"{home_frame}")

        print(f"推定時刻     : " f"{home['time_sec']:.6f} sec")

        print(f"方法         : " f"{home['method']}")

        if home["before_frame"] is not None:

            print(f"前Frame      : " f"{home['before_frame']}")

        if home["after_frame"] is not None:

            print(f"後Frame      : " f"{home['after_frame']}")

        print()
        print("------------------------------------------")

        print("速度解析")

        print("------------------------------------------")

        print(f"飛行時間      : " f"{flight_time:.6f} sec")

        print(f"投球距離      : " f"{PITCH_DISTANCE_M:.3f} m")

        print(f"速度          : " f"{speed_mps:.3f} m/s")

        print(f"速度          : " f"{speed_kmh:.2f} km/h")

        print("------------------------------------------")

        # ----------------------------------------------------
        # 結果
        # ----------------------------------------------------

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "home_frame": home_frame,
            "release_time_sec": release["time_sec"],
            "home_time_sec": home["time_sec"],
            "flight_time_sec": flight_time,
            "pitch_distance_m": PITCH_DISTANCE_M,
            "speed_mps": speed_mps,
            "speed_kmh": speed_kmh,
            "release_method": release["method"],
            "home_method": home["method"],
            "release_before_frame": release["before_frame"],
            "release_after_frame": release["after_frame"],
            "home_before_frame": home["before_frame"],
            "home_after_frame": home["after_frame"],
            "valid_points": len(rows),
        }

        output_csv = OUTPUT_DIR / f"{video_name}_v12_1.csv"

        save_result(output_csv, result)

        print()
        print(f"CSV: {output_csv}")

        summary.append(result)

    # ========================================================
    # まとめCSV
    # ========================================================

    summary_csv = OUTPUT_DIR / "pitch_speed_v12_1_summary.csv"

    fields = [
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
        "release_before_frame",
        "release_after_frame",
        "home_before_frame",
        "home_after_frame",
        "valid_points",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fields)

        writer.writeheader()
        writer.writerows(summary)

    # ========================================================
    # 最終表示
    # ========================================================

    print()
    print("=" * 60)
    print("v12.1 全動画解析完了")
    print("=" * 60)

    for result in summary:

        print(
            f"{result['video']} | "
            f"{result['speed_kmh']:.2f} km/h | "
            f"飛行時間 "
            f"{result['flight_time_sec']:.6f} sec | "
            f"Release "
            f"{result['release_method']} | "
            f"Home "
            f"{result['home_method']}"
        )

    print()
    print(f"まとめCSV: {summary_csv}")

    print(f"出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
