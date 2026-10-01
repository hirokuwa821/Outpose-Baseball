"""
pitch_speed_v22_auto_distance.py

投球速度解析 v22 (左右投手対応・姿勢推定×ボール解析 改良版)
- プレート位置取得フレームを固定 (PLATE_REF_FRAME) にして、左投手でも安定した座標を取得
- ネットの Y 座標は自動検出（キーポイント全体の最大 Y）
- エクステンションが不自然に大きい場合は警告し、距離を 0 以上にクランプ
- HANDEDNESS 辞書で動画ごとに左投 (True) / 右投 (False) を指定可能
"""

import csv
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 設定
# ============================================================

# 入出力ディレクトリ
INPUT_DIR = Path("output/ball_tracking_v5")
KEYPOINTS_PATH = Path("output/keypoints.csv")
OUTPUT_DIR = Path("output/pitch_speed_v22")

# 投球場全長（プレートからネットまで）の実測距離 (メートル)
TOTAL_FIELD_DISTANCE_M = 18.0

# プレート位置取得に使うフレーム番号（動画開始直後は姿勢が不安定なため）
PLATE_REF_FRAME = 15

# ------------------------------------------------------------
# 動画ごとのリリースフレーム・ネットフレーム（手動で決めたもの）
# ------------------------------------------------------------
BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 92),
    "pitch_98_2": (66, 86),
}

# ------------------------------------------------------------
# 投手の左右情報（True=左投げ, False=右投げ）
# ------------------------------------------------------------
HANDEDNESS = {
    "pitch_102_1": True,
    "pitch_98_1": True,
    "pitch_98_2": True,
}

# 空気抵抗パラメータ (既存と同じ)
CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

# ============================================================
# ユーティリティ関数
# ============================================================

def get_foot_coordinates(df: pd.DataFrame, release_frame: int, is_left_handed: bool):
    """COCO キーポイント (足首) から支え足・踏み出し足の Y 座標を取得。

    - 右投げ (is_left_handed=False): 支え足=右足首(16), 踏み出し足=左足首(15)
    - 左投げ (is_left_handed=True):  支え足=左足首(15), 踏み出し足=右足首(16)
    """
    stance_idx = 15 if is_left_handed else 16   # 支え足（プレートに近い足）
    stride_idx = 16 if is_left_handed else 15   # 踏み出し足（投球時に前に出す足）
    try:
        # プレート位置は安定フレーム PLATE_REF_FRAME で支え足の Y を取得
        y_plate = float(df[df['frame'] == PLATE_REF_FRAME][f'y_{stance_idx}'].values[0])
        # リリース時の踏み出し足 Y 座標
        y_foot = float(df[df['frame'] == release_frame][f'y_{stride_idx}'].values[0])
        return y_plate, y_foot
    except Exception as e:
        print(f"[Error] 足座標取得失敗 (frame={release_frame}, left={is_left_handed}): {e}")
        return None, None

def detect_net_y(df: pd.DataFrame) -> float:
    """キーポイント全体の最大 Y 座標をネット（地面）位置とみなす。"""
    y_cols = [c for c in df.columns if c.startswith('y_')]
    return float(df[y_cols].max().max())

def calculate_dynamic_distance(y_plate: float, y_net: float, y_foot: float, total_distance: float):
    pixel_total = abs(y_plate - y_net)
    if pixel_total == 0:
        return total_distance, 0.0
    pixel_ext = abs(y_plate - y_foot)
    extension_m = total_distance * (pixel_ext / pixel_total)
    # 80% 以上のエクステンションは異常とみなす
    if extension_m > 0.8 * total_distance:
        print(f"[Warning] 推定エクステンションが大きすぎます ({extension_m:.2f} m)。" )
    pitch_distance = max(total_distance - extension_m, 0.0)  # 負になるのは防ぐ
    return pitch_distance, extension_m

def load_ball_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                conf = float(r.get('confidence', 0))
                if conf < 0.3:
                    continue
                rows.append({
                    'frame': int(r['frame']),
                    'time_sec': float(r['time_sec']),
                    'cx': float(r['cx']),
                    'cy': float(r['cy'])
                })
            except Exception:
                continue
    rows.sort(key=lambda x: x['frame'])
    return rows

# ============================================================
# メイン処理
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("投球速度解析 v22 (左右投手対応 改良版)")
    print("=" * 60)

    if not KEYPOINTS_PATH.exists():
        print(f"[Error] キーポイントファイルが見つかりません: {KEYPOINTS_PATH}")
        return
    kp_df = pd.read_csv(KEYPOINTS_PATH)
    y_net = detect_net_y(kp_df)
    print(f"[Info] 推定ネット Y 座標: {y_net:.1f} ピクセル")

    for video_name, (release_frame, net_frame) in BASE_FRAMES.items():
        is_left = HANDEDNESS.get(video_name, False)
        y_plate, y_foot = get_foot_coordinates(kp_df, release_frame, is_left)
        if y_plate is None:
            continue

        pitch_distance, extension = calculate_dynamic_distance(
            y_plate, y_net, y_foot, TOTAL_FIELD_DISTANCE_M
        )

        ball_csv = INPUT_DIR / f"{video_name}_tracking.csv"
        if not ball_csv.exists():
            print(f"[Warning] ボールトラッキング CSV が見つかりません: {ball_csv}")
            continue
        rows = load_ball_csv(ball_csv)
        if len(rows) < 2:
            print(f"[Warning] データ不足: {video_name}")
            continue
        frames = np.array([r['frame'] for r in rows])
        times = np.array([r['time_sec'] for r in rows])
        # タイムスタンプ平滑化（一次関数）
        time_model = np.poly1d(np.polyfit(frames, times, 1))
        flight_time = time_model(net_frame) - time_model(release_frame)
        if flight_time <= 0:
            print(f"[Warning] 飛行時間が負です: {video_name}")
            continue

        release_speed_mps = (math.exp(K_DRAG * pitch_distance) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        hand_str = "左投" if is_left else "右投"
        print(f"動画: {video_name}")
        print(f"  投手: {hand_str}")
        print(f"  AI取得座標: プレートY={y_plate:.1f}, 踏み出し足Y={y_foot:.1f}")
        print(f"  エクステンション: {extension:.2f} m   投球距離: {pitch_distance:.2f} m")
        print(f"  飛行時間: {flight_time:.4f} s   初速推定: {release_speed_kmh:.2f} km/h")
        print("-" * 60)

if __name__ == "__main__":
    main()

