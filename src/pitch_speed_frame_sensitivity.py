"""
投球速度 フレーム感度解析
リリース・ホーム到達フレームを±2フレーム動かして速度を比較

投球距離:
18.44 m - 1.95 m = 16.49 m
"""

import csv
from pathlib import Path
import cv2

# ============================================================
# 設定
# ============================================================

VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("output/pitch_speed_frame_sensitivity")

# 現在採用している投球距離
PITCH_DISTANCE_M = 16.49

# 現在の基準フレーム
BASE_FRAMES = {
    "pitch_00015_80.mp4": (429, 455),
    "pitch_102_1.mp4": (51, 70),
    "pitch_98_1.mp4": (72, 94),
    "pitch_98_2.mp4": (66, 88),
}

# 基準フレームから前後何フレーム調べるか
OFFSET = 2


# ============================================================
# FPS取得
# ============================================================


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, total_frames


# ============================================================
# 速度計算
# ============================================================


def calculate_speed(release_frame, home_frame, fps):
    frame_diff = home_frame - release_frame

    if frame_diff <= 0:
        return None

    flight_time = frame_diff / fps

    speed_mps = PITCH_DISTANCE_M / flight_time

    speed_kmh = speed_mps * 3.6

    return flight_time, speed_kmh


# ============================================================
# メイン
# ============================================================


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("投球速度 フレーム感度解析")
    print("=" * 60)

    print(f"投球距離: " f"{PITCH_DISTANCE_M:.3f} m")

    print(f"フレーム変動: " f"±{OFFSET} frame")

    print("=" * 60)

    all_results = []

    for video_name, (base_release, base_home) in BASE_FRAMES.items():

        print("\n")
        print("=" * 60)
        print(f"動画: {video_name}")
        print("=" * 60)

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():

            print(f"動画が見つかりません: " f"{video_path}")

            continue

        fps, total_frames = get_video_info(video_path)

        print(f"FPS: {fps:.4f}")

        print(f"総フレーム数: " f"{total_frames}")

        print(f"基準Release: " f"Frame {base_release}")

        print(f"基準Home: " f"Frame {base_home}")

        results = []

        # ----------------------------------------------------
        # ±2フレームを全組み合わせで計算
        # ----------------------------------------------------

        for release_offset in range(-OFFSET, OFFSET + 1):

            release_frame = base_release + release_offset

            for home_offset in range(-OFFSET, OFFSET + 1):

                home_frame = base_home + home_offset

                if release_frame < 0:
                    continue

                if home_frame >= total_frames:
                    continue

                result = calculate_speed(release_frame, home_frame, fps)

                if result is None:
                    continue

                flight_time, speed_kmh = result

                data = {
                    "video": video_name,
                    "release_frame": release_frame,
                    "home_frame": home_frame,
                    "release_offset": release_offset,
                    "home_offset": home_offset,
                    "frame_diff": home_frame - release_frame,
                    "flight_time_sec": flight_time,
                    "speed_kmh": speed_kmh,
                }

                results.append(data)
                all_results.append(data)

        # ----------------------------------------------------
        # CSV保存
        # ----------------------------------------------------

        csv_path = OUTPUT_DIR / (Path(video_name).stem + "_frame_sensitivity.csv")

        fieldnames = [
            "video",
            "release_frame",
            "home_frame",
            "release_offset",
            "home_offset",
            "frame_diff",
            "flight_time_sec",
            "speed_kmh",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:

            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        # ----------------------------------------------------
        # 速度一覧
        # ----------------------------------------------------

        print("\n")
        print("------------------------------------------")
        print("速度一覧")
        print("------------------------------------------")

        for release_frame in range(base_release - OFFSET, base_release + OFFSET + 1):

            row = [r for r in results if r["release_frame"] == release_frame]

            row.sort(key=lambda x: x["home_frame"])

            if not row:
                continue

            text = []

            for r in row:

                text.append(f"H{r['home_frame']}: " f"{r['speed_kmh']:.2f}")

            print(f"Release {release_frame}: " + " | ".join(text) + " km/h")

        # ----------------------------------------------------
        # 基準値
        # ----------------------------------------------------

        base_result = None

        for r in results:

            if r["release_frame"] == base_release and r["home_frame"] == base_home:

                base_result = r
                break

        if base_result:

            print("\n")
            print("------------------------------------------")
            print("基準値")
            print("------------------------------------------")

            print(f"Release " f"{base_release}" f" → Home " f"{base_home}")

            print(f"飛行時間: " f"{base_result['flight_time_sec']:.5f} sec")

            print(f"速度: " f"{base_result['speed_kmh']:.2f} km/h")

        # ----------------------------------------------------
        # 最低・最高
        # ----------------------------------------------------

        if results:

            min_result = min(results, key=lambda x: x["speed_kmh"])

            max_result = max(results, key=lambda x: x["speed_kmh"])

            print("\n")
            print("------------------------------------------")
            print("フレーム選択による速度範囲")
            print("------------------------------------------")

            print(f"最低: " f"{min_result['speed_kmh']:.2f} km/h")

            print(
                f"  "
                f"Release "
                f"{min_result['release_frame']}"
                f" → Home "
                f"{min_result['home_frame']}"
            )

            print(f"最高: " f"{max_result['speed_kmh']:.2f} km/h")

            print(
                f"  "
                f"Release "
                f"{max_result['release_frame']}"
                f" → Home "
                f"{max_result['home_frame']}"
            )

        print("\n")
        print(f"CSV: {csv_path}")

    # ========================================================
    # 全動画まとめCSV
    # ========================================================

    summary_path = OUTPUT_DIR / "frame_sensitivity_summary.csv"

    fieldnames = [
        "video",
        "release_frame",
        "home_frame",
        "release_offset",
        "home_offset",
        "frame_diff",
        "flight_time_sec",
        "speed_kmh",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(all_results)

    print("\n")
    print("=" * 60)
    print("フレーム感度解析完了")
    print("=" * 60)

    print(f"まとめCSV: " f"{summary_path}")

    print(f"出力先: " f"{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
