"""
pitch_speed_v17_final_manual.py

投球速度解析 v17 (最終手動調整版)
- AIによる不確定な自動検知を排除し、目視フレームを最優先する
- 時間の平滑化(1次関数)のみを行い、タイムスタンプのブレを除去
- ネット投球を想定し、投球距離を 17.0m に設定
"""

import csv
import math
import numpy as np
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v17")

# アマチュアの平均的なエクステンションを考慮したネットまでの距離
PITCH_DISTANCE_M = 17.0  
MIN_CONFIDENCE = 0.3

# ★ ここをコマ送り動画を見て微修正してください！ ★
# (リリースフレーム, ネットに触れた瞬間のフレーム)
BASE_FRAMES = {
    "pitch_102_1": (51, 70),  
    "pitch_98_1": (72, 94),   # 例: AIの予測では 91 または 92 付近が衝突の瞬間でした
    "pitch_98_2": (66, 88),
}

# 物理ドラグ定数（空気抵抗）
CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

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
                cx = float(row["cx"])
                cy = float(row["cy"])
                conf = float(row["confidence"])
                if conf >= MIN_CONFIDENCE:
                    rows.append({"frame": frame, "time_sec": time_sec, "cx": cx, "cy": cy})
            except:
                pass
    rows.sort(key=lambda x: x["frame"])
    return rows

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
    print("投球速度解析 v17 (最終手動調整版)")
    print(f"想定投球距離: {PITCH_DISTANCE_M} m")
    print("=" * 60)

    summary = []

    for video_name, (release_frame, net_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            print(f"スキップ: {csv_path} が存在しません。")
            continue

        rows = load_csv(csv_path)
        if len(rows) < 2:
            continue

        frames = np.array([r["frame"] for r in rows])
        times = np.array([r["time_sec"] for r in rows])

        # 1. タイムスタンプの平滑化 (1次関数: time = a * frame + b)
        # これにより動画の微小なコマ落ちなどのノイズを無効化します
        time_model = np.poly1d(np.polyfit(frames, times, 1))

        # 手動指定フレームの正確な時間を取得
        release_time = time_model(release_frame)
        net_time = time_model(net_frame)

        flight_time = net_time - release_time
        if flight_time <= 0:
            print(f"スキップ: {video_name} 飛行時間が0秒以下です。フレーム指定を確認してください。")
            continue

        # 速度計算
        avg_speed_mps = PITCH_DISTANCE_M / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6

        # 空気抵抗補正 (v0)
        release_speed_mps = (math.exp(K_DRAG * PITCH_DISTANCE_M) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "net_frame": net_frame,
            "flight_time_sec": flight_time,
            "avg_speed_kmh": avg_speed_kmh,
            "release_speed_kmh": release_speed_kmh,
        }
        
        output_csv = OUTPUT_DIR / f"{video_name}_v17.csv"
        save_result(output_csv, result)
        summary.append(result)

    if summary:
        print("\n解析結果 (v17)")
        print("-" * 60)
        for res in summary:
            print(f"動画: {res['video']}")
            print(f"  指定フレーム: リリース {res['release_frame']} -> ネット {res['net_frame']}")
            print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒")
            print(f"  初速推定: {res['release_speed_kmh']:.2f} km/h")
            print("-" * 60)

if __name__ == "__main__":
    main()

