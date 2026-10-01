"""
pitch_speed_v13_drag_correction.py

投球速度解析 v13
空気抵抗（抗力）モデルを導入した初速（リリース速度）の逆算補正版

数式モデル:
- 減速度 a = - k * v^2
- k = 0.5 * Cd * rho * A / m
  - Cd (抗力係数) = 0.30 (標準的な野球ボールの値)
  - rho (空気密度) = 1.204 kg/m^3 (常温1気圧)
  - A (断面積) = pi * (0.073 / 2)^2 = 0.004185 m^2 (公式球直径 73mm)
  - m (質量) = 0.145 kg (公式球重量 145g)
  - 補正係数 k ≈ 0.0052 m^-1
- 積分方程式より、リリース初速 v0 は以下のように求まります:
  v0 = (exp(k * D) - 1) / (k * T)
  (ここで D = 投球距離, T = 飛行時間)
"""

import csv
import math
from pathlib import Path

# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/pitch_speed_v13")

PITCH_DISTANCE_M = 16.49  # プレートからリリース前方距離を引いた距離
MIN_CONFIDENCE = 0.3

# 手動で指定した基準フレーム (v12.1と同じ)
BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 94),
    "pitch_98_2": (66, 88),
}

# ============================================================
# 物理ドラグ定数
# ============================================================
CD = 0.30          # 抗力係数
RHO = 1.204        # 空気密度 (kg/m^3)
DIAMETER = 0.073   # ボールの直径 (m)
MASS = 0.145       # ボールの質量 (kg)

AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS  # 約 0.005216 m^-1

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
# 指定フレームの時刻を推定
# ============================================================

def estimate_time(rows, target_frame):
    if not rows:
        return None

    # 完全一致
    for row in rows:
        if row["frame"] == target_frame:
            return {
                "time_sec": row["time_sec"],
                "frame_float": float(target_frame),
                "method": "exact",
                "before_frame": target_frame,
                "after_frame": target_frame,
            }

    # 前後の追跡点を探す
    before = None
    after = None
    for row in rows:
        if row["frame"] < target_frame:
            before = row
        elif row["frame"] > target_frame:
            after = row
            break

    # 前後両方がある場合
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

    # 指定フレームが追跡範囲より前の場合
    if before is None:
        first = rows[0]
        return {
            "time_sec": first["time_sec"],
            "frame_float": float(first["frame"]),
            "method": "clamped_before",
            "before_frame": None,
            "after_frame": first["frame"],
        }

    # 指定フレームが追跡範囲より後の場合
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
        "avg_speed_mps",
        "avg_speed_kmh",
        "release_speed_mps",
        "release_speed_kmh",
        "release_method",
        "home_method",
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
    print("投球速度解析 v13 (空気抵抗モデル補正版)")
    print("=" * 60)
    print(f"投球距離: {PITCH_DISTANCE_M:.3f} m")
    print(f"ドラグ定数 k: {K_DRAG:.6f} m^-1")
    print("=" * 60)

    summary = []

    for video_name, (release_frame, home_frame) in BASE_FRAMES.items():
        csv_path = INPUT_DIR / f"{video_name}_tracking.csv"
        if not csv_path.exists():
            print(f"CSV {csv_path} が見つかりません。スキップします。")
            continue

        rows = load_csv(csv_path)
        if len(rows) < 2:
            print(f"{video_name}: 追跡点が不足しています。")
            continue

        release = estimate_time(rows, release_frame)
        home = estimate_time(rows, home_frame)

        if release is None or home is None:
            print(f"{video_name}: 時刻推定に失敗しました。")
            continue

        flight_time = home["time_sec"] - release["time_sec"]
        if flight_time <= 0:
            print(f"{video_name}: 飛行時間が0以下です。")
            continue

        # 1. 単純平均速度
        avg_speed_mps = PITCH_DISTANCE_M / flight_time
        avg_speed_kmh = avg_speed_mps * 3.6

        # 2. 空気抵抗を考慮したリリース初速 (v0)
        # v0 = (exp(k * D) - 1) / (k * T)
        release_speed_mps = (math.exp(K_DRAG * PITCH_DISTANCE_M) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        result = {
            "video": video_name,
            "release_frame": release_frame,
            "home_frame": home_frame,
            "release_time_sec": release["time_sec"],
            "home_time_sec": home["time_sec"],
            "flight_time_sec": flight_time,
            "pitch_distance_m": PITCH_DISTANCE_M,
            "avg_speed_mps": avg_speed_mps,
            "avg_speed_kmh": avg_speed_kmh,
            "release_speed_mps": release_speed_mps,
            "release_speed_kmh": release_speed_kmh,
            "release_method": release["method"],
            "home_method": home["method"],
            "valid_points": len(rows),
        }

        output_csv = OUTPUT_DIR / f"{video_name}_v13.csv"
        save_result(output_csv, result)
        summary.append(result)

    # まとめCSV保存
    summary_csv = OUTPUT_DIR / "pitch_speed_v13_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        fields = list(summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    # 最終表示
    print("\n" + "=" * 60)
    print("解析結果比較 (平均速度 vs リリース初速)")
    print("=" * 60)
    for res in summary:
        print(f"動画: {res['video']}")
        print(f"  飛行時間: {res['flight_time_sec']:.4f} 秒")
        print(f"  平均速度: {res['avg_speed_kmh']:.2f} km/h")
        print(f"  初速推定: {res['release_speed_kmh']:.2f} km/h  (空気抵抗補正: +{res['release_speed_kmh'] - res['avg_speed_kmh']:.2f} km/h)")
        print("-" * 60)

    print(f"\nまとめCSVを保存しました: {summary_csv}")

if __name__ == "__main__":
    main()
