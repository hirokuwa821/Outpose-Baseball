"""
pitch_speed_v15_subframe.py

投球速度解析 v15
- 曲線フィッティングによるY座標からのサブフレーム（小数時刻）逆算
- マイル（mph）出力の追加
- スローモーション映像補正機能の追加
"""

import csv
import math
import numpy as np
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v15")

PITCH_DISTANCE_M = 17.0  # エクステンションを約1.44m（アマチュア平均程度）と想定
MIN_CONFIDENCE = 0.3

# 動画がスローモーションの場合の補正（例: 2倍スローなら 2.0 に設定）
VIDEO_SPEED_RATIO = 1.0

BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 94),
    "pitch_98_2": (66, 88),
}

CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

KMH_TO_MPH = 1.0 / 1.60934  # km/h を mph に変換する係数

# ============================================================
# 関数
# ============================================================

def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row["frame"])
                time_sec = float(row["time_sec"])
                cy = float(row["cy"])
                conf = float(row["confidence"])
                if conf >= MIN_CONFIDENCE:
                    rows.append({"frame": frame, "time_sec": time_sec, "cy": cy})
            except:
                pass
    rows.sort(key=lambda x: x["frame"])
    return rows

def solve_quadratic_time(poly_model, target_y, hint_time):
    """
    at^2 + bt + c = target_y となる t を解き、hint_timeに一番近い解を返す
    """
    coeffs = poly_model.coeffs.copy()
    coeffs[-1] -= target_y
    roots = np.roots(coeffs)
    # 実数解のみ取得
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        return hint_time
    # 予想される時間 (hint_time) に最も近い解を採用
    best_root = min(real_roots, key=lambda r: abs(r - hint_time))
    return best_root

def get_y_for_frame(rows, target_frame):
    """指定フレームのY座標を取得（無い場合は一番近いフレームのもの）"""
    closest_row = min(rows, key=lambda r: abs(r["frame"] - target_frame))
    return closest_row["cy"]

def save_result(path, result):
    fields = list(result.keys())
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
    print("投球速度解析 v15 (サブフレーム逆算 & スロー補正版)")
    print(f"スローモーション補正係数: {VIDEO_SPEED_RATIO} 倍")
    print("=" * 60)

    summary = []

    for video_name, (release_frame, home_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            continue

        rows = load_csv(csv_path)
        if len(rows) < 3:
            continue

        frames = np.array([r["frame"] for r in rows])
        
        # 補正係数を使って現実のタイムスタンプに修正
        times = np.array([r["time_sec"] / VIDEO_SPEED_RATIO for r in rows])
        cys = np.array([r["cy"] for r in rows])

        # 1. タイムスタンプモデル
        time_model = np.poly1d(np.polyfit(frames, times, 1))
        smoothed_times = time_model(frames)

        # 2. 軌道モデル (cy = at^2 + bt + c)
        traj_model = np.poly1d(np.polyfit(smoothed_times, cys, 2))

        # 3. Y座標の基準ラインを取得
        release_y = get_y_for_frame(rows, release_frame)
        home_y = get_y_for_frame(rows, home_frame)

        # 4. Y座標から「正確な小数の時刻」を逆算
        hint_release_time = time_model(release_frame)
        hint_home_time = time_model(home_frame)

        refined_release_time = solve_quadratic_time(traj_model, release_y, hint_release_time)
        refined_home_time = solve_quadratic_time(traj_model, home_y, hint_home_time)

        flight_time = refined_home_time - refined_release_time
        if flight_time <= 0:
            continue

        # 速度計算
        avg_speed_mps = PITCH_DISTANCE_M / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6
        avg_speed_mph = avg_speed_kmh * KMH_TO_MPH

        # 空気抵抗補正初速 (v0)
        release_speed_mps = (math.exp(K_DRAG * PITCH_DISTANCE_M) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6
        release_speed_mph = release_speed_kmh * KMH_TO_MPH

        result = {
            "video": video_name,
            "flight_time_sec": flight_time,
            "avg_speed_kmh": avg_speed_kmh,
            "avg_speed_mph": avg_speed_mph,
            "release_speed_kmh": release_speed_kmh,
            "release_speed_mph": release_speed_mph,
        }
        
        output_csv = OUTPUT_DIR / f"{video_name}_v15.csv"
        save_result(output_csv, result)
        summary.append(result)

    if summary:
        print("\n解析結果 (v15)")
        print("-" * 60)
        for res in summary:
            print(f"動画: {res['video']}")
            print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒")
            print(f"  平均速度: {res['avg_speed_mph']:.2f} mph ({res['avg_speed_kmh']:.1f} km/h)")
            print(f"  初速推定: {res['release_speed_mph']:.2f} mph ({res['release_speed_kmh']:.1f} km/h)")
            print("-" * 60)

if __name__ == "__main__":
    main()

