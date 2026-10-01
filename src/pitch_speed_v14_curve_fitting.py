"""
pitch_speed_v14_curve_fitting.py

投球速度解析 v14
曲線フィッティング（多項式回帰）を利用してタイムスタンプと軌道のノイズを平滑化するバージョン
"""

import csv
import math
import numpy as np
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v14")

PITCH_DISTANCE_M = 16.49  # プレートからリリース前方距離を引いた距離
MIN_CONFIDENCE = 0.3

# 手動で指定した基準フレーム
BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 94),
    "pitch_98_2": (66, 88),
}

# 物理ドラグ定数
CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

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

            rows.append({
                "frame": frame,
                "time_sec": time_sec,
                "cx": cx,
                "cy": cy,
                "confidence": confidence,
            })
    rows.sort(key=lambda x: x["frame"])
    return rows

# ============================================================
# 個別結果保存
# ============================================================

def save_result(path, result):
    fields = [
        "video", "release_frame", "home_frame",
        "release_time_sec", "home_time_sec", "flight_time_sec",
        "pitch_distance_m", "avg_speed_kmh", "release_speed_kmh",
        "valid_points"
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
    print("投球速度解析 v14 (曲線フィッティング平滑化版)")
    print("=" * 60)

    summary = []

    for video_name, (release_frame, home_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            print(f"CSV {csv_path} が見つかりません。")
            continue

        rows = load_csv(csv_path)
        if len(rows) < 3:
            print(f"{video_name}: 追跡点が不足しています（フィッティングには最低3点必要）。")
            continue

        # データ配列の作成
        frames = np.array([r["frame"] for r in rows])
        times = np.array([r["time_sec"] for r in rows])
        cys = np.array([r["cy"] for r in rows])

        # 1. タイムスタンプの平滑化 (1次関数: time = a * frame + b)
        # 動画のコマ落ちやタイムスタンプのブレを吸収して滑らかな時間を算出
        time_model = np.poly1d(np.polyfit(frames, times, 1))

        # 2. Y座標（縦の軌道）の平滑化 (2次関数: cy = a * time^2 + b * time + c)
        # トラッキングのブレを放物線で滑らかにする（将来的に座標からの時刻逆算に使える基盤）
        smoothed_times = time_model(frames)
        trajectory_model_y = np.poly1d(np.polyfit(smoothed_times, cys, 2))

        # 平滑化されたモデルを使って、リリースとホーム到達の時刻を推定
        release_time = time_model(release_frame)
        home_time = time_model(home_frame)

        flight_time = home_time - release_time
        if flight_time <= 0:
            print(f"{video_name}: 飛行時間が0以下です。")
            continue

        # 速度計算
        avg_speed_mps = PITCH_DISTANCE_M / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6

        # 空気抵抗補正 (v13と同等)
        release_speed_mps = (math.exp(K_DRAG * PITCH_DISTANCE_M) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "home_frame": home_frame,
            "release_time_sec": release_time,
            "home_time_sec": home_time,
            "flight_time_sec": flight_time,
            "pitch_distance_m": PITCH_DISTANCE_M,
            "avg_speed_kmh": avg_speed_kmh,
            "release_speed_kmh": release_speed_kmh,
            "valid_points": len(rows),
        }

        output_csv = OUTPUT_DIR / f"{video_name}_v14.csv"
        save_result(output_csv, result)
        summary.append(result)

    # まとめCSV保存
    if summary:
        summary_csv = OUTPUT_DIR / "pitch_speed_v14_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            fields = list(summary[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

        print("\n解析結果比較 (v14 フィッティング平滑化版)")
        print("-" * 60)
        for res in summary:
            print(f"動画: {res['video']}")
            print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒 (平滑化モデル)")
            print(f"  初速推定: {res['release_speed_kmh']:.2f} km/h")
            print("-" * 60)

if __name__ == "__main__":
    main()

