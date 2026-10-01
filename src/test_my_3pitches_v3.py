from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 設定
# ============================================================

VIDEOS = [
    ("videos/pitch_102_1.mp4", 102.0),
    ("videos/pitch_98_1.mp4", 98.0),
    ("videos/pitch_98_2.mp4", 98.0),
]

MODEL_FILE = Path("output/train839_feature_selection/best_train839_speed_model.pkl")

OUTPUT_DIR = Path("output/my_3pitches_test_v3")
TRACKING_DIR = OUTPUT_DIR / "tracking"


# ============================================================
# TOP_20特徴量
# ============================================================

FEATURES = [
    "start_x",
    "std_dy",
    "path_straightness",
    "median_step_distance",
    "max_pixel_speed",
    "average_pixel_speed",
    "frame_start",
    "mean_step_distance",
    "frame_end",
    "end_x",
    "x_range",
    "std_pixel_speed",
    "start_end_distance_px",
    "total_distance_px",
    "start_y",
    "total_dx",
    "max_step_distance",
    "mean_dx",
    "std_dx",
    "trajectory_angle_deg",
]


# ============================================================
# 特徴量計算
# ============================================================


def make_features(csv_file):

    df = pd.read_csv(csv_file)

    if len(df) < 2:
        raise ValueError(f"追跡点が少なすぎます: {len(df)}")

    x = df["center_x"].astype(float).to_numpy()
    y = df["center_y"].astype(float).to_numpy()

    frame = df["frame"].astype(float).to_numpy()

    dx = np.diff(x)
    dy = np.diff(y)

    distance = np.sqrt(dx**2 + dy**2)

    frame_diff = np.diff(frame)

    frame_diff[frame_diff == 0] = 1

    pixel_speed = distance / frame_diff

    total_dx = x[-1] - x[0]
    total_dy = y[-1] - y[0]

    total_distance = np.sum(distance)

    start_end_distance = np.sqrt(total_dx**2 + total_dy**2)

    if total_distance > 0:
        path_straightness = start_end_distance / total_distance
    else:
        path_straightness = 0.0

    angle = np.degrees(np.arctan2(total_dy, total_dx))

    features = {
        "start_x": x[0],
        "std_dy": np.std(dy),
        "path_straightness": path_straightness,
        "median_step_distance": np.median(distance),
        "max_pixel_speed": np.max(pixel_speed),
        "average_pixel_speed": np.mean(pixel_speed),
        "frame_start": frame[0],
        "mean_step_distance": np.mean(distance),
        "frame_end": frame[-1],
        "end_x": x[-1],
        "x_range": np.max(x) - np.min(x),
        "std_pixel_speed": np.std(pixel_speed),
        "start_end_distance_px": start_end_distance,
        "total_distance_px": total_distance,
        "start_y": y[0],
        "total_dx": total_dx,
        "max_step_distance": np.max(distance),
        "mean_dx": np.mean(dx),
        "std_dx": np.std(dx),
        "trajectory_angle_deg": angle,
    }

    return features


# ============================================================
# メイン処理
# ============================================================


def main():

    print("=" * 60)
    print("自分の3投球 TOP_20モデル検証 v3")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # モデル読み込み
    # --------------------------------------------------------

    print()
    print("モデル:")
    print(MODEL_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"モデルがありません: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)

    print("モデル読み込み完了")

    # --------------------------------------------------------
    # モデル特徴量確認
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("モデル特徴量")
    print("=" * 60)

    for i, feature in enumerate(FEATURES, 1):
        print(f"{i:2d}. {feature}")

    # --------------------------------------------------------
    # 動画ごとに追跡
    # --------------------------------------------------------

    results = []
    feature_rows = []

    for i, (video, actual_speed) in enumerate(VIDEOS, 1):

        video_path = Path(video)

        print()
        print("#" * 60)
        print(f"投球 {i}/3")
        print("#" * 60)

        print()
        print("動画:", video_path)
        print("実測:", actual_speed, "km/h")

        if not video_path.exists():
            print("ERROR: 動画がありません")
            continue

        tracking_csv = TRACKING_DIR / f"pitch_{i}_tracking.csv"

        # ----------------------------------------------------
        # 追跡CSVを作成
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("ボール追跡")
        print("-" * 60)

        print("入力:", video_path)
        print("出力:", tracking_csv)

        command = [
            sys.executable,
            "src/track_ball.py",
            str(video_path),
        ]

        print()
        print("実行:")
        print(" ".join(command))

        # 重要：
        # text=True を使わず、UTF-8/CP932問題を避ける
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        )

        output = process.stdout.decode("utf-8", errors="replace")

        print(output)

        # ----------------------------------------------------
        # track_ball.py が生成した
        # output/ball_tracking.csv を取得
        # ----------------------------------------------------

        source_csv = Path("output/ball_tracking.csv")

        if not source_csv.exists():
            print("ERROR: ball_tracking.csv が作成されませんでした")
            continue

        # 別名保存
        tracking_df = pd.read_csv(source_csv)

        tracking_df.to_csv(tracking_csv, index=False, encoding="utf-8-sig")

        print()
        print("追跡CSV保存:")
        print(tracking_csv)

        print("追跡点数:", len(tracking_df))

        # ----------------------------------------------------
        # CSV内容確認
        # ----------------------------------------------------

        if len(tracking_df) < 2:
            print("ERROR: 追跡点が少なすぎます")
            continue

        # ----------------------------------------------------
        # 特徴量作成
        # ----------------------------------------------------

        features = make_features(tracking_csv)

        print()
        print("-" * 60)
        print("特徴量")
        print("-" * 60)

        for feature in FEATURES:
            print(f"{feature:25s}: " f"{features[feature]:.6f}")

        # ----------------------------------------------------
        # 予測
        # ----------------------------------------------------

        X = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)

        predicted_speed = float(model.predict(X)[0])

        error = predicted_speed - actual_speed
        absolute_error = abs(error)

        print()
        print("-" * 60)
        print("推定結果")
        print("-" * 60)

        print(f"実測球速 : {actual_speed:.2f} km/h")

        print(f"推定球速 : {predicted_speed:.2f} km/h")

        print(f"誤差     : {error:+.2f} km/h")

        print(f"絶対誤差 : {absolute_error:.2f} km/h")

        # ----------------------------------------------------
        # 保存用
        # ----------------------------------------------------

        result = {
            "video": str(video_path),
            "actual_speed_kmh": actual_speed,
            "predicted_speed_kmh": predicted_speed,
            "error_kmh": error,
            "absolute_error_kmh": absolute_error,
            "tracking_points": len(tracking_df),
        }

        results.append(result)

        feature_row = {
            "video": str(video_path),
            "actual_speed_kmh": actual_speed,
        }

        feature_row.update(features)

        feature_rows.append(feature_row)

    # ========================================================
    # 追跡結果がない場合
    # ========================================================

    if len(results) == 0:
        print()
        print("3本とも処理できませんでした。")
        return

    # ========================================================
    # 結果保存
    # ========================================================

    results_df = pd.DataFrame(results)

    predictions_file = OUTPUT_DIR / "my_3pitches_predictions_v3.csv"

    results_df.to_csv(predictions_file, index=False, encoding="utf-8-sig")

    features_df = pd.DataFrame(feature_rows)

    features_file = OUTPUT_DIR / "my_3pitches_features_v3.csv"

    features_df.to_csv(features_file, index=False, encoding="utf-8-sig")

    # ========================================================
    # 同一追跡チェック
    # ========================================================

    print()
    print("=" * 60)
    print("追跡結果の一致チェック")
    print("=" * 60)

    csv_files = sorted(TRACKING_DIR.glob("pitch_*_tracking.csv"))

    tracking_data = []

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        tracking_data.append(
            df[
                [
                    "frame",
                    "time_sec",
                    "center_x",
                    "center_y",
                    "confidence",
                ]
            ].copy()
        )

    if len(tracking_data) >= 2:

        for i in range(len(tracking_data)):
            for j in range(i + 1, len(tracking_data)):

                same = tracking_data[i].equals(tracking_data[j])

                print(f"投球{i+1} vs 投球{j+1}:", same)

    # ========================================================
    # 総合評価
    # ========================================================

    actual = results_df["actual_speed_kmh"]

    predicted = results_df["predicted_speed_kmh"]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mean_error = np.mean(predicted - actual)

    print()
    print("=" * 60)
    print("3本総合")
    print("=" * 60)

    print(f"MAE      : {mae:.3f} km/h")

    print(f"RMSE     : {rmse:.3f} km/h")

    print(f"平均誤差 : {mean_error:+.3f} km/h")

    # ========================================================
    # 結果表示
    # ========================================================

    print()
    print("=" * 60)
    print("最終結果")
    print("=" * 60)

    print(
        results_df[
            [
                "video",
                "actual_speed_kmh",
                "predicted_speed_kmh",
                "error_kmh",
                "absolute_error_kmh",
                "tracking_points",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 60)
    print("保存先")
    print("=" * 60)

    print("予測結果:")
    print(predictions_file)

    print()
    print("特徴量:")
    print(features_file)

    print()
    print("追跡CSV:")
    print(TRACKING_DIR)

    print()
    print("=" * 60)
    print("検証完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
