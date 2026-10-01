"""
pitch_speed_v16_net_throw.py

投球速度解析 v16 (ネット投球特化版)
- ボールの速度変化（加速度）からネットへの衝突瞬間を自動検知
- 衝突後のノイズデータを除外したクリーンな時間計算
"""

import csv
import math
import numpy as np
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v6_kalman")
OUTPUT_DIR = Path("output/pitch_speed_v16_kalman")

PITCH_DISTANCE_M = 17.0  # リリース位置からネットまでの距離
MIN_CONFIDENCE = 0.3

# 手動で指定した「リリース」フレームのみ使用
# （ホーム到達フレームはAIが自動検知するため不要ですが、一応辞書形式を保ちます）
BASE_FRAMES = {
    "pitch_102_1": 51,
    "pitch_98_1": 72,
    "pitch_98_2": 66,
}

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
                cy = float(row["cy"])
                conf = float(row["confidence"])
                if conf >= MIN_CONFIDENCE:
                    rows.append({"frame": frame, "time_sec": time_sec, "cy": cy})
            except:
                pass
    rows.sort(key=lambda x: x["frame"])
    return rows

def detect_net_impact(rows, release_frame):
    """
    リリース以降のボールのY方向の移動速度を計算し、
    速度が激減（または跳ね返りでマイナスに）なったフレームをネット衝突とみなす
    """
    valid_rows = [r for r in rows if r["frame"] >= release_frame]
    if len(valid_rows) < 3:
        return None

    # フレーム間の移動速度(ピクセル/秒)を計算
    velocities = []
    for i in range(1, len(valid_rows)):
        dy = valid_rows[i]["cy"] - valid_rows[i-1]["cy"]
        dt = valid_rows[i]["time_sec"] - valid_rows[i-1]["time_sec"]
        if dt > 0:
            velocities.append((valid_rows[i]["frame"], dy / dt))

    # ボールは通常、下(プラス方向)に向かって等速〜やや加速で動く
    # ネットに当たると、速度が急激に落ちるかマイナスになる
    impact_frame = valid_rows[-1]["frame"] # デフォルトは最後のフレーム
    
    for i in range(1, len(velocities)):
        curr_frame, curr_v = velocities[i]
        prev_frame, prev_v = velocities[i-1]
        
        # リリース直後（10フレーム以内）は、手のブレやトラッキングの迷いで
        # 速度が不安定になるため、衝突判定から除外（保護期間）
        if curr_frame < release_frame + 10:
            continue
            
        # 前のフレームに比べて速度が 50% 以下に落ちた、あるいはマイナスになったら衝突と判定
        if curr_v < (prev_v * 0.5) or curr_v < 0:
            impact_frame = curr_frame
            break

    return impact_frame

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
    print("投球速度解析 v16 (ネット投球特化版)")
    print(f"想定投球距離: {PITCH_DISTANCE_M} m")
    print("=" * 60)

    summary = []

    for video_name, release_frame in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            print(f"スキップ: {csv_path} が存在しません。(apply_kalman_filter.py は実行しましたか？)")
            continue

        rows = load_csv(csv_path)
        if len(rows) < 3:
            print(f"スキップ: {video_name} のトラッキングデータが不足しています。")
            continue

        # 1. ネット衝突フレームの自動検知
        impact_frame = detect_net_impact(rows, release_frame)
        if not impact_frame:
            print(f"スキップ: {video_name} ネット衝突フレームが検知できませんでした。")
            continue

        # 2. 衝突後ノイズを除外した「クリーンな空中データ」の抽出
        # 衝突フレーム自体も歪んでいる可能性があるので、その手前までのデータを平滑化に使う
        clean_rows = [r for r in rows if release_frame <= r["frame"] <= impact_frame]
        
        if len(clean_rows) < 2:
            continue

        frames = np.array([r["frame"] for r in clean_rows])
        times = np.array([r["time_sec"] for r in clean_rows])

        # 3. タイムスタンプの平滑化（クリーンなデータのみを使用）
        time_model = np.poly1d(np.polyfit(frames, times, 1))

        # 平滑化モデルを使って時刻を算出
        release_time = time_model(release_frame)
        impact_time = time_model(impact_frame)

        flight_time = impact_time - release_time
        if flight_time <= 0:
            continue

        # 速度計算
        avg_speed_mps = PITCH_DISTANCE_M / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6

        # 空気抵抗補正初速 (v0)
        release_speed_mps = (math.exp(K_DRAG * PITCH_DISTANCE_M) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "auto_impact_frame": impact_frame,
            "flight_time_sec": flight_time,
            "avg_speed_kmh": avg_speed_kmh,
            "release_speed_kmh": release_speed_kmh,
        }
        
        output_csv = OUTPUT_DIR / f"{video_name}_v16.csv"
        save_result(output_csv, result)
        summary.append(result)

    if summary:
        print("\n解析結果 (v16)")
        print("-" * 60)
        for res in summary:
            print(f"動画: {res['video']}")
            print(f"  リリースフレーム: {res['release_frame']} / ネット到達(自動検知): {res['auto_impact_frame']}")
            print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒")
            print(f"  初速推定: {res['release_speed_kmh']:.2f} km/h")
            print("-" * 60)

if __name__ == "__main__":
    main()
