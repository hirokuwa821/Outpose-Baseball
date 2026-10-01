"""
pitch_speed_v19_auto_distance.py

投球速度解析 v19 (完全自動化・融合版)
- 姿勢推定(keypoints.csv)からピッチャーの足の座標を自動取得
- ボールトラッキングAIと融合し、距離と球速を全自動で計算するアプリのプロトタイプ
"""

import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
KEYPOINTS_PATH = Path("output/keypoints.csv")  # 姿勢推定の出力結果
OUTPUT_DIR = Path("output/pitch_speed_v19")

TOTAL_FIELD_DISTANCE_M = 18.0  # プレートからネットまでの現実の距離

# ★ ここはアプリ化の際に画像認識（白線検知など）で自動化するか、
# 一度だけユーザーに画面下部をタップしてもらう仕様にします
Y_NET_PIXEL = 300  

BASE_FRAMES = {
    "pitch_102_1": (51, 70),  
    "pitch_98_1": (72, 92),   
    "pitch_98_2": (66, 86),
}

CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

# ============================================================
# 骨格データからの座標自動取得モジュール
# ============================================================

def get_foot_coordinates(keypoints_df, release_frame, is_lefty=False):
    """
    keypoints.csv から、プレート位置（軸足）と踏み出し位置（前足）のY座標を自動取得する。
    COCO format: 左足首=15, 右足首=16
    """
    pivot_idx = 15 if is_lefty else 16  # 軸足
    stride_idx = 16 if is_lefty else 15 # 踏み出し足

    try:
        # 1. プレート位置 (y_plate)
        # モーション開始時（例としてフレーム0）の軸足のY座標
        frame0_data = keypoints_df[keypoints_df['frame'] == 0]
        y_plate = frame0_data[f'y_{pivot_idx}'].values[0]

        # 2. 踏み出し位置 (y_foot)
        # リリース時の踏み出し足のY座標
        release_data = keypoints_df[keypoints_df['frame'] == release_frame]
        y_foot = release_data[f'y_{stride_idx}'].values[0]
        
        return float(y_plate), float(y_foot)
        
    except Exception as e:
        print(f"骨格座標の取得に失敗しました: {e}")
        return None, None

# ============================================================
# 距離計算・ボール解析モジュール
# ============================================================

def calculate_dynamic_distance(y_plate, y_net, y_foot, total_distance):
    pixel_total = abs(y_plate - y_net)
    if pixel_total == 0: return total_distance, 0.0
    pixel_ext = abs(y_plate - y_foot)
    extension_m = total_distance * (pixel_ext / pixel_total)
    return total_distance - extension_m, extension_m

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
                if float(row["confidence"]) >= 0.3:
                    rows.append({"frame": frame, "time_sec": time_sec, "cx": cx, "cy": cy})
            except: pass
    rows.sort(key=lambda x: x["frame"])
    return rows

# ============================================================
# メイン (アプリのバックエンド挙動)
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("投球速度解析 v19 (姿勢推定×ボール解析 融合版)")
    print("=" * 60)

    # 姿勢推定データ（keypoints.csv）をロード
    if not KEYPOINTS_PATH.exists():
        print(f"エラー: {KEYPOINTS_PATH} が見つかりません。")
        return
        
    keypoints_df = pd.read_csv(KEYPOINTS_PATH)

    for video_name, (release_frame, net_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists(): continue

        # ① 【AI連携】骨格データからピッチャーの足の座標を全自動取得！
        y_plate, y_foot = get_foot_coordinates(keypoints_df, release_frame, is_lefty=False)
        if y_plate is None: continue

        # ② 【距離計算】取得した足の座標から、投球距離を動的計算！
        pitch_distance, extension = calculate_dynamic_distance(
            y_plate, Y_NET_PIXEL, y_foot, TOTAL_FIELD_DISTANCE_M
        )

        # ③ 【ボール解析】平滑化して正確な時間を出し、初速を計算！
        rows = load_csv(csv_path)
        if len(rows) < 2: continue

        frames = np.array([r["frame"] for r in rows])
        times = np.array([r["time_sec"] for r in rows])
        time_model = np.poly1d(np.polyfit(frames, times, 1))
        flight_time = time_model(net_frame) - time_model(release_frame)
        
        if flight_time <= 0: continue

        avg_speed_kmh = (pitch_distance / flight_time) * 3.6
        release_speed_kmh = ((math.exp(K_DRAG * pitch_distance) - 1) / (K_DRAG * flight_time)) * 3.6

        print(f"動画: {video_name}")
        print(f"  AI取得座標: プレートY={y_plate:.1f}, 踏み出し足Y={y_foot:.1f}")
        print(f"  自動計算距離: エクステンション {extension:.2f}m (投球距離 {pitch_distance:.2f}m)")
        print(f"  初速推定: {release_speed_kmh:.2f} km/h")
        print("-" * 60)

if __name__ == "__main__":
    main()

