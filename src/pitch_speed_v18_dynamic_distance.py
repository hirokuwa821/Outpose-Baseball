"""
pitch_speed_v18_dynamic_distance.py

投球速度解析 v18 (動的エクステンション計算版)
- 映像内の「プレート」「ネット」「踏み出し足」のY座標から、投球距離を動的に計算する
- アプリ化に向けた、距離固定(17.0m)からの脱却
"""

import csv
import math
import numpy as np
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v18")
MIN_CONFIDENCE = 0.3

# プレートからネット（またはキャッチャー）までの現実の総距離(m)
# 例: プレートからネットまで 18.0m 確保している場合
TOTAL_FIELD_DISTANCE_M = 18.0  

# ★ アプリ化時のUI想定 ★
# ユーザーが画面上で「プレート」「ネット下部」「踏み出し足」をタップして設定するY座標(ピクセル)
# ※ここでは仮のピクセル座標を入れています。ご自身の動画の座標に書き換えてください。
USER_TAPS = {
    "pitch_102_1": {"y_plate": 900, "y_net": 300, "y_foot": 750},
    "pitch_98_1":  {"y_plate": 950, "y_net": 320, "y_foot": 800},
    "pitch_98_2":  {"y_plate": 950, "y_net": 320, "y_foot": 780},
}

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
# 関数
# ============================================================

def calculate_dynamic_distance(y_plate, y_net, y_foot, total_distance):
    """
    画像上のY座標から、実際のエクステンション（踏み出し距離）と投球距離を計算する。
    ※カメラが遠くズームされている場合は線形比率で近似可能。
    """
    # 画面上での全体のピクセル距離
    pixel_total = abs(y_plate - y_net)
    if pixel_total == 0:
        return total_distance, 0.0
    
    # プレートから踏み出し足までのピクセル距離
    pixel_ext = abs(y_plate - y_foot)
    
    # 割合から現実のエクステンション(m)を計算
    extension_m = total_distance * (pixel_ext / pixel_total)
    
    # 実際の投球距離(m) = 総距離 - エクステンション
    actual_pitch_distance = total_distance - extension_m
    return actual_pitch_distance, extension_m

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
    print("投球速度解析 v18 (動的エクステンション計算版)")
    print("=" * 60)

    summary = []

    for video_name, (release_frame, net_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            continue

        # タップ座標の取得
        taps = USER_TAPS.get(video_name)
        if not taps:
            continue

        # 動的距離計算
        pitch_distance, extension = calculate_dynamic_distance(
            taps["y_plate"], taps["y_net"], taps["y_foot"], TOTAL_FIELD_DISTANCE_M
        )

        rows = load_csv(csv_path)
        if len(rows) < 2:
            continue

        frames = np.array([r["frame"] for r in rows])
        times = np.array([r["time_sec"] for r in rows])

        time_model = np.poly1d(np.polyfit(frames, times, 1))
        flight_time = time_model(net_frame) - time_model(release_frame)
        
        if flight_time <= 0:
            continue

        # 計算された動的な投球距離を使用して速度を計算
        avg_speed_mps = pitch_distance / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6
        release_speed_mps = (math.exp(K_DRAG * pitch_distance) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        result = {
            "video": video_name,
            "flight_time_sec": flight_time,
            "extension_m": extension,
            "pitch_distance_m": pitch_distance,
            "release_speed_kmh": release_speed_kmh,
        }
        
        output_csv = OUTPUT_DIR / f"{video_name}_v18.csv"
        save_result(output_csv, result)
        summary.append(result)

    if summary:
        print("\n解析結果 (v18)")
        print("-" * 60)
        for res in summary:
            print(f"動画: {res['video']}")
            print(f"  踏み出し幅(エクステンション): {res['extension_m']:.2f} m")
            print(f"  実際の投球距離: {res['pitch_distance_m']:.2f} m")
            print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒")
            print(f"  初速推定: {res['release_speed_kmh']:.2f} km/h")
            print("-" * 60)

if __name__ == "__main__":
    main()

