import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 設定
# ==========================================

INPUT_FILE = "output/ball_tracking.csv"
OUTPUT_FILE = "output/ball_features.csv"

FPS = 29.97002997


# ==========================================
# CSV読み込み
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("==========================================")
print("ボール軌跡の特徴量作成")
print("==========================================")
print()

print("入力データ:")
print(df)
print()


# ==========================================
# データ確認
# ==========================================

if len(df) < 2:
    print("ボール検出フレームが少なすぎます。")
    exit()


# ==========================================
# 基本情報
# ==========================================

frame_start = int(df["frame"].iloc[0])
frame_end = int(df["frame"].iloc[-1])

num_frames = len(df)

duration = df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]


# ==========================================
# 座標
# ==========================================

x = df["center_x"].to_numpy()
y = df["center_y"].to_numpy()


# ==========================================
# フレーム間の移動量
# ==========================================

dx = np.diff(x)
dy = np.diff(y)

distance_per_frame = np.sqrt(dx**2 + dy**2)


# ==========================================
# 総移動距離
# ==========================================

total_distance = np.sum(distance_per_frame)


# ==========================================
# 始点から終点までの直線距離
# ==========================================

start_end_distance = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)


# ==========================================
# 平均速度（画面上）
# ==========================================

if duration > 0:

    average_pixel_speed = total_distance / duration

else:

    average_pixel_speed = 0


# ==========================================
# 最大速度（フレーム間）
# ==========================================

if len(distance_per_frame) > 0:

    max_pixel_speed = np.max(distance_per_frame) * FPS

else:

    max_pixel_speed = 0


# ==========================================
# X方向・Y方向の移動量
# ==========================================

total_dx = x[-1] - x[0]
total_dy = y[-1] - y[0]


# ==========================================
# 軌跡の曲がり具合
# ==========================================

if len(distance_per_frame) >= 2:

    directions = np.arctan2(dy, dx)

    direction_change = np.diff(directions)

    # -π～πに補正
    direction_change = (direction_change + np.pi) % (2 * np.pi) - np.pi

    total_direction_change = np.sum(np.abs(direction_change))

else:

    total_direction_change = 0


# ==========================================
# 平均信頼度
# ==========================================

if "confidence" in df.columns:

    mean_confidence = df["confidence"].mean()

    min_confidence = df["confidence"].min()

else:

    mean_confidence = 0

    min_confidence = 0


# ==========================================
# 特徴量を1行にまとめる
# ==========================================

features = {
    "frame_start": frame_start,
    "frame_end": frame_end,
    "num_frames": num_frames,
    "duration": duration,
    "total_distance_px": total_distance,
    "start_end_distance_px": start_end_distance,
    "average_pixel_speed": average_pixel_speed,
    "max_pixel_speed": max_pixel_speed,
    "total_dx": total_dx,
    "total_dy": total_dy,
    "direction_change": total_direction_change,
    "mean_confidence": mean_confidence,
    "min_confidence": min_confidence,
}


features_df = pd.DataFrame([features])


# ==========================================
# 保存
# ==========================================

Path("output").mkdir(parents=True, exist_ok=True)

features_df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# 結果表示
# ==========================================

print()
print("==========================================")
print("特徴量作成完了")
print("==========================================")
print()

print(features_df.T)

print()
print("保存先:")
print(OUTPUT_FILE)
