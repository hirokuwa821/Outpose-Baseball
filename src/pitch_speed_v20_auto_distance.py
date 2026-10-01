"""
pitch_speed_v20_auto_distance.py

投球速度解析 v20 (左右投手対応・姿勢推定×ボール解析 融合版)
- ユーザーが右投げか左投げかを選択できるようにし、足のインデックスを自動で切り替えます。
- ネット位置の Y 座標は画像上の地面とみなす固定ピクセル（Y_NET_PIXEL）ですが、左投でも同じです。
- 既存の v19 スクリプトをベースに、handedness マッピングとインデックス選択ロジックを追加しました。
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
KEYPOINTS_PATH = Path("output/keypoints.csv")  # 姿勢推定CSV（COCOフォーマット）
OUTPUT_DIR = Path("output/pitch_speed_v20")

# プレートからネットまでの実測総距離（メートル）
TOTAL_FIELD_DISTANCE_M = 18.0

# ネットの地面の Y 座標（画像上のピクセル）
# アプリ化時は UI でユーザーにタップさせても OK
Y_NET_PIXEL = 300

# -----------------------------------------------------------------
# 1) 動画ごとのリリースフレーム・ネットフレーム（手動で決めたもの）
# -----------------------------------------------------------------
BASE_FRAMES = {
    "pitch_102_1": (51, 70),
    "pitch_98_1": (72, 92),
    "pitch_98_2": (66, 86),
}

# -----------------------------------------------------------------
# 2) 投手の利き手設定（右投げ: False, 左投げ: True）
#    必要なら動画ごとに個別設定できるように dict を用意
# -----------------------------------------------------------------
HANDEDNESS = {
    "pitch_102_1": False,  # 右投げ（例）
    "pitch_98_1": False,
    "pitch_98_2": False,
    # 例: 左投げの場合は True に設定します
    # "some_lefty_video": True,
}

# 空気抵抗定数（従来と同じ）
CD = 0.30
RHO = 1.204
DIAMETER = 0.073
MASS = 0.145
AREA = math.pi * (DIAMETER / 2) ** 2
K_DRAG = (0.5 * CD * RHO * AREA) / MASS

# ============================================================
# 姿勢データから足の座標を取得するユーティリティ
# ============================================================

def get_foot_coordinates(df: pd.DataFrame, release_frame: int, is_left_handed: bool):
    """COCO keypoints の足首インデックスを利き手に合わせて取得。

    - 右投げ (is_left_handed=False):
        * スタンス（支え足） = 右足首 (index 16)
        * ストライド（踏み出し足） = 左足首 (index 15)
    - 左投げ (is_left_handed=True):
        * スタンス = 左足首 (index 15)
        * ストライド = 右足首 (index 16)
    """
    # COCO の足首インデックスは 15 (左足首) と 16 (右足首)
    stance_idx = 15 if is_left_handed else 16   # 支え足
    stride_idx = 16 if is_left_handed else 15   # 踏み出し足

    try:
        # プレート位置はモーション開始時（frame=0）に支え足の Y 座標を使用
        plate_row = df[df['frame'] == 0]
        y_plate = float(plate_row[f'y_{stance_idx}'].values[0])

        # リリース時の踏み出し足の Y 座標
        release_row = df[df['frame'] == release_frame]
        y_foot = float(release_row[f'y_{stride_idx}'].values[0])
        return y_plate, y_foot
    except Exception as e:
        print(f"[Error] 足座標取得失敗 (frame={release_frame}, left_handed={is_left_handed}): {e}")
        return None, None

# ============================================================
# 動的エクステンション計算
# ============================================================

def calculate_dynamic_distance(y_plate: float, y_net: float, y_foot: float, total_distance: float):
    pixel_total = abs(y_plate - y_net)
    if pixel_total == 0:
        return total_distance, 0.0
    pixel_ext = abs(y_plate - y_foot)
    extension_m = total_distance * (pixel_ext / pixel_total)
    pitch_distance = total_distance - extension_m
    return pitch_distance, extension_m

# ============================================================
# ボールトラッキング CSV ローダー（共通）
# ============================================================

def load_ball_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                conf = float(r["confidence"]) if "confidence" in r else 0.0
                if conf < 0.3:
                    continue
                rows.append({
                    "frame": int(r["frame"]),
                    "time_sec": float(r["time_sec"]),
                    "cx": float(r["cx"]),
                    "cy": float(r["cy"]),
                })
            except Exception:
                continue
    rows.sort(key=lambda x: x["frame"])
    return rows

# ============================================================
# メイン処理
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("投球速度解析 v20 (左右投手対応 融合版)")
    print("=" * 60)

    if not KEYPOINTS_PATH.exists():
        print(f"[Error] キーポイントファイルが見つかりません: {KEYPOINTS_PATH}")
        return
    keypoints_df = pd.read_csv(KEYPOINTS_PATH)

    for video_name, (release_frame, net_frame) in BASE_FRAMES.items():
        # --- 1) 足座標取得 -------------------------------------------------
        is_left = HANDEDNESS.get(video_name, False)  # デフォルトは右投げ
        y_plate, y_foot = get_foot_coordinates(keypoints_df, release_frame, is_left)
        if y_plate is None:
            continue

        # --- 2) 動的投球距離計算 -------------------------------------------
        pitch_distance, extension = calculate_dynamic_distance(
            y_plate, Y_NET_PIXEL, y_foot, TOTAL_FIELD_DISTANCE_M
        )

        # --- 3) ボールトラッキング解析 --------------------------------------
        ball_csv = INPUT_DIR / f"{video_name}_tracking.csv"
        if not ball_csv.exists():
            print(f"[Warning] ボールトラッキングCSV が見つかりません: {ball_csv}")
            continue
        rows = load_ball_csv(ball_csv)
        if len(rows) < 2:
            print(f"[Warning] データ不足: {video_name}")
            continue
        frames = np.array([r["frame"] for r in rows])
        times = np.array([r["time_sec"] for r in rows])
        # 1次関数でタイムスタンプ平滑化（v14 と同様）
        time_model = np.poly1d(np.polyfit(frames, times, 1))
        flight_time = time_model(net_frame) - time_model(release_frame)
        if flight_time <= 0:
            print(f"[Warning] 飛行時間が負です: {video_name}")
            continue

        # 初速推定（空気抵抗補正）
        release_speed_mps = (math.exp(K_DRAG * pitch_distance) - 1) / (K_DRAG * flight_time)
        release_speed_kmh = release_speed_mps * 3.6

        # --- 4) 結果表示 -------------------------------------------------
        print(f"動画: {video_name}")
        hand_str = "左投" if is_left else "右投"
        print(f"  投手: {hand_str}")
        print(f"  AI取得座標: プレートY={y_plate:.1f}, 踏み出し足Y={y_foot:.1f}")
        print(f"  エクステンション: {extension:.2f} m   投球距離: {pitch_distance:.2f} m")
        print(f"  飛行時間: {flight_time:.4f} s   初速推定: {release_speed_kmh:.2f} km/h")
        print("-" * 60)

if __name__ == "__main__":
    main()

