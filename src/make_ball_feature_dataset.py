import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 設定
# ==========================================

# train-11で再追跡した33動画
TRACKING_DIR = Path("output/train11_tracking_33")

# train-11版の特徴量データ
OUTPUT_FILE = Path("output/ball_feature_dataset_train11.csv")


# ==========================================
# 特徴量を計算
# ==========================================


def calculate_features(csv_path):

    df = pd.read_csv(csv_path)

    if len(df) < 2:
        return None

    x = df["center_x"].to_numpy()
    y = df["center_y"].to_numpy()

    confidence = df["confidence"].to_numpy()

    # --------------------------------------
    # フレーム
    # --------------------------------------

    frame_start = int(df["frame"].iloc[0])

    frame_end = int(df["frame"].iloc[-1])

    num_frames = len(df)

    # --------------------------------------
    # 時間
    # --------------------------------------

    duration = df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]

    if duration <= 0:
        return None

    # --------------------------------------
    # 1フレームごとの移動量
    # --------------------------------------

    dx = np.diff(x)
    dy = np.diff(y)

    distance = np.sqrt(dx**2 + dy**2)

    # --------------------------------------
    # 総移動距離
    # --------------------------------------

    total_distance_px = np.sum(distance)

    # --------------------------------------
    # 始点から終点までの直線距離
    # --------------------------------------

    start_end_distance_px = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

    # --------------------------------------
    # 平均ピクセル速度
    # --------------------------------------

    average_pixel_speed = total_distance_px / duration

    # --------------------------------------
    # 最大フレーム間速度
    # --------------------------------------

    frame_times = np.diff(df["time_sec"].to_numpy())

    valid = frame_times > 0

    if np.any(valid):

        frame_speeds = distance[valid] / frame_times[valid]

        max_pixel_speed = np.max(frame_speeds)

    else:

        max_pixel_speed = 0

    # --------------------------------------
    # X方向・Y方向の移動
    # --------------------------------------

    total_dx = x[-1] - x[0]

    total_dy = y[-1] - y[0]

    # --------------------------------------
    # 方向変化
    # --------------------------------------

    if len(dx) >= 2:

        angles = np.arctan2(dy, dx)

        angle_diff = np.diff(np.unwrap(angles))

        direction_change = np.sum(np.abs(angle_diff))

    else:

        direction_change = 0

    # --------------------------------------
    # 信頼度
    # --------------------------------------

    mean_confidence = np.mean(confidence)

    min_confidence = np.min(confidence)

    # --------------------------------------
    # 結果
    # --------------------------------------

    return {
        "video": csv_path.stem.replace("_tracking", ""),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "num_frames": num_frames,
        "duration": duration,
        "total_distance_px": total_distance_px,
        "start_end_distance_px": start_end_distance_px,
        "average_pixel_speed": average_pixel_speed,
        "max_pixel_speed": max_pixel_speed,
        "total_dx": total_dx,
        "total_dy": total_dy,
        "direction_change": direction_change,
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
    }


# ==========================================
# 全CSV処理
# ==========================================


def main():

    csv_files = sorted(TRACKING_DIR.glob("*_tracking.csv"))

    print("追跡CSV:", len(csv_files), "本")

    if len(csv_files) != 33:

        print()
        print("警告: 追跡CSVが33本ではありません")

        print("現在:", len(csv_files))

        return

    results = []

    for csv_path in csv_files:

        print()
        print("解析:", csv_path.name)

        features = calculate_features(csv_path)

        if features is None:

            print("特徴量を作成できません")

            continue

        results.append(features)

    # ======================================
    # 保存
    # ======================================

    if not results:

        print("特徴量を作成できませんでした")

        return

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ======================================
    # 結果
    # ======================================

    print()
    print("=" * 60)
    print("train-11特徴量データセット作成完了")
    print("=" * 60)

    print()

    print("動画数:", len(result_df))

    print()

    print(result_df.to_string(index=False))

    print()

    print("保存先:", OUTPUT_FILE)


# ==========================================
# 実行
# ==========================================

if __name__ == "__main__":
    main()
