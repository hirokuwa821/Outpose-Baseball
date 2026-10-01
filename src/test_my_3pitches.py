from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys
import re

# ============================================================
# 設定
# ============================================================

MODEL_PATH = Path("output/train839_feature_selection/best_train839_speed_model.pkl")

# あなたの3本
VIDEOS = [
    {
        "path": Path("videos/pitch_102_1.mp4"),
        "actual_speed": 102.0,
    },
    {
        "path": Path("videos/pitch_98_1.mp4"),
        "actual_speed": 98.0,
    },
    {
        "path": Path("videos/pitch_98_2.mp4"),
        "actual_speed": 98.0,
    },
]

OUTPUT_DIR = Path("output/my_3pitches_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 839本モデルのTOP_20
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
# 表示
# ============================================================

print("=" * 60)
print("自分の3投球動画 839本モデル検証")
print("=" * 60)

print()
print("モデル:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"モデルがありません: {MODEL_PATH}")


# ============================================================
# モデル読み込み
# ============================================================

print()
print("=" * 60)
print("モデル読み込み")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# 動画確認
# ============================================================

print()
print("=" * 60)
print("動画確認")
print("=" * 60)

for item in VIDEOS:

    path = item["path"]

    print()
    print("動画:", path)
    print("実測:", item["actual_speed"], "km/h")

    if not path.exists():
        raise FileNotFoundError(f"動画がありません: {path}")


# ============================================================
# 既存のボール追跡スクリプトを利用
# ============================================================


def run_tracking(video_path, output_csv):
    """
    既存のtrack_ball.pyを利用して
    1本の動画からball_tracking.csvを作成する。

    ※ 実際のtrack_ball.pyの引数仕様が異なる場合は
       この部分だけ調整する。
    """

    print()
    print("-" * 60)
    print("ボール追跡")
    print("-" * 60)

    print("入力:", video_path)
    print("出力:", output_csv)

    command = [
        sys.executable,
        "src/track_ball.py",
        str(video_path),
    ]

    print()
    print("実行:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)

        raise RuntimeError(f"ボール追跡に失敗しました: {video_path}")

    # 既存track_ball.pyがoutput/ball_tracking.csvを作る場合
    default_csv = Path("output/ball_tracking.csv")

    if not default_csv.exists():
        raise FileNotFoundError("追跡CSVが作成されませんでした: " + str(default_csv))

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    default_csv.replace(output_csv)

    print()
    print("追跡CSV保存:", output_csv)


# ============================================================
# CSVから特徴量作成
# ============================================================


def make_features(csv_path):

    df = pd.read_csv(csv_path)

    print()
    print("追跡データ:", len(df), "行")

    # --------------------------------------------------------
    # 列名確認
    # --------------------------------------------------------

    print()
    print("CSV列:")
    print(list(df.columns))

    # --------------------------------------------------------
    # 座標列を探す
    # --------------------------------------------------------

    x_candidates = [
        "x",
        "ball_x",
        "center_x",
    ]

    y_candidates = [
        "y",
        "ball_y",
        "center_y",
    ]

    x_col = None
    y_col = None

    for col in x_candidates:
        if col in df.columns:
            x_col = col
            break

    for col in y_candidates:
        if col in df.columns:
            y_col = col
            break

    if x_col is None or y_col is None:

        raise ValueError(
            "追跡CSVからx,y座標を見つけられませんでした。\n" f"列: {list(df.columns)}"
        )

    print()
    print("使用座標:")
    print("X:", x_col)
    print("Y:", y_col)

    # --------------------------------------------------------
    # 数値化
    # --------------------------------------------------------

    df[x_col] = pd.to_numeric(
        df[x_col],
        errors="coerce",
    )

    df[y_col] = pd.to_numeric(
        df[y_col],
        errors="coerce",
    )

    df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)

    if len(df) < 3:

        raise ValueError("有効な追跡点が少なすぎます: " + str(len(df)))

    # --------------------------------------------------------
    # 座標
    # --------------------------------------------------------

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)

    step_distance = np.sqrt(dx**2 + dy**2)

    # --------------------------------------------------------
    # フレーム
    # --------------------------------------------------------

    frame_col = None

    for col in [
        "frame",
        "frame_id",
        "frame_index",
    ]:
        if col in df.columns:
            frame_col = col
            break

    if frame_col is not None:

        frames = pd.to_numeric(
            df[frame_col],
            errors="coerce",
        ).to_numpy()

        valid_frames = frames[np.isfinite(frames)]

        if len(valid_frames) > 0:

            frame_start = float(valid_frames[0])

            frame_end = float(valid_frames[-1])

            num_frames = len(valid_frames)

        else:

            frame_start = 0
            frame_end = len(df) - 1
            num_frames = len(df)

    else:

        frame_start = 0
        frame_end = len(df) - 1
        num_frames = len(df)

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = 30.0

    # --------------------------------------------------------
    # 基本特徴量
    # --------------------------------------------------------

    total_dx = x[-1] - x[0]
    total_dy = y[-1] - y[0]

    total_distance_px = np.sum(step_distance)

    start_end_distance_px = np.sqrt(total_dx**2 + total_dy**2)

    average_pixel_speed = total_distance_px / max((len(df) - 1) / fps, 1e-8)

    max_pixel_speed = np.max(step_distance) * fps

    mean_step_distance = np.mean(step_distance)

    std_step_distance = np.std(step_distance)

    max_step_distance = np.max(step_distance)

    median_step_distance = np.median(step_distance)

    # --------------------------------------------------------
    # 直線性
    # --------------------------------------------------------

    if total_distance_px > 0:

        path_straightness = start_end_distance_px / total_distance_px

    else:

        path_straightness = 0.0

    # --------------------------------------------------------
    # 軌跡角度
    # --------------------------------------------------------

    trajectory_angle_deg = np.degrees(
        np.arctan2(
            total_dy,
            total_dx,
        )
    )

    # --------------------------------------------------------
    # 平均・標準偏差
    # --------------------------------------------------------

    mean_dx = np.mean(dx)
    mean_dy = np.mean(dy)

    std_dx = np.std(dx)
    std_dy = np.std(dy)

    pixel_speed = step_distance * fps

    std_pixel_speed = np.std(pixel_speed)

    duration_sec = max(frame_end - frame_start, 1) / fps

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    features = {
        "start_x": x[0],
        "std_dy": std_dy,
        "path_straightness": path_straightness,
        "median_step_distance": median_step_distance,
        "max_pixel_speed": max_pixel_speed,
        "average_pixel_speed": average_pixel_speed,
        "frame_start": frame_start,
        "mean_step_distance": mean_step_distance,
        "frame_end": frame_end,
        "end_x": x[-1],
        "x_range": np.max(x) - np.min(x),
        "std_pixel_speed": std_pixel_speed,
        "start_end_distance_px": start_end_distance_px,
        "total_distance_px": total_distance_px,
        "start_y": y[0],
        "total_dx": total_dx,
        "max_step_distance": max_step_distance,
        "mean_dx": mean_dx,
        "std_dx": std_dx,
        "trajectory_angle_deg": trajectory_angle_deg,
    }

    result = pd.DataFrame([features])

    return result


# ============================================================
# 3本を処理
# ============================================================

results = []

print()
print("=" * 60)
print("3本の動画を検証")
print("=" * 60)

for i, item in enumerate(VIDEOS, start=1):

    video_path = item["path"]
    actual_speed = item["actual_speed"]

    print()
    print("#" * 60)
    print(f"動画 {i}/3")
    print("#" * 60)

    print("動画:", video_path)
    print("実測:", actual_speed, "km/h")

    # --------------------------------------------------------
    # 追跡CSV
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / f"pitch_{i}_tracking.csv"

    run_tracking(
        video_path,
        csv_path,
    )

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    feature_df = make_features(csv_path)

    # 必要特徴量確認
    missing = [f for f in FEATURES if f not in feature_df.columns]

    if missing:

        raise ValueError("必要特徴量がありません:\n" + "\n".join(missing))

    X = feature_df[FEATURES].copy()

    # --------------------------------------------------------
    # 推定
    # --------------------------------------------------------

    prediction = float(model.predict(X)[0])

    error = prediction - actual_speed

    absolute_error = abs(error)

    print()
    print("=" * 60)
    print("推定結果")
    print("=" * 60)

    print()
    print(f"実測球速 : {actual_speed:.2f} km/h")
    print(f"推定球速 : {prediction:.2f} km/h")
    print(f"誤差     : {error:+.2f} km/h")
    print(f"絶対誤差 : {absolute_error:.2f} km/h")

    row = {
        "video": str(video_path),
        "actual_speed_kmh": actual_speed,
        "predicted_speed_kmh": prediction,
        "error_kmh": error,
        "absolute_error_kmh": absolute_error,
    }

    for feature in FEATURES:
        row[feature] = float(feature_df.iloc[0][feature])

    results.append(row)


# ============================================================
# 結果保存
# ============================================================

result_df = pd.DataFrame(results)

output_csv = OUTPUT_DIR / "my_3pitches_predictions.csv"

result_df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# MAE
# ============================================================

mae = np.mean(result_df["absolute_error_kmh"])

rmse = np.sqrt(np.mean(result_df["error_kmh"] ** 2))

mean_error = np.mean(result_df["error_kmh"])


# ============================================================
# 最終結果
# ============================================================

print()
print()
print("=" * 60)
print("3本の未知データ検証結果")
print("=" * 60)

print()

for _, row in result_df.iterrows():

    print(f"{Path(row['video']).name}")

    print(f"  実測 : " f"{row['actual_speed_kmh']:.2f} km/h")

    print(f"  推定 : " f"{row['predicted_speed_kmh']:.2f} km/h")

    print(f"  誤差 : " f"{row['error_kmh']:+.2f} km/h")

    print()


print("=" * 60)
print("総合結果")
print("=" * 60)

print()
print(f"MAE       : {mae:.3f} km/h")

print(f"RMSE      : {rmse:.3f} km/h")

print(f"平均誤差  : {mean_error:+.3f} km/h")

print()

print("結果保存:")
print(output_csv)

print()
print("=" * 60)
print("検証完了")
print("=" * 60)
