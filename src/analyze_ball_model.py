from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 設定
# ==========================================

BASE_DIR = Path.cwd()

DATASET_FILE = BASE_DIR / "output" / "ball_tracking_dataset_40.csv"
PREDICTION_FILE = BASE_DIR / "output" / "extratrees_ball_speed_predictions_40.csv"


# ==========================================
# データ読み込み
# ==========================================

print("=" * 60)
print("球速予測モデルの診断")
print("=" * 60)

print("\nデータ読み込み中...")

dataset = pd.read_csv(DATASET_FILE)
pred = pd.read_csv(PREDICTION_FILE)

print("特徴量データ:", len(dataset), "件")
print("予測データ:", len(pred), "件")


# ==========================================
# videoをキーに結合
# ==========================================

df = dataset.merge(
    pred[["video", "actual_speed_kmh", "predicted_speed_kmh", "error_kmh"]],
    on="video",
    how="inner",
    suffixes=("", "_pred"),
)

print("結合後:", len(df), "件")


# ==========================================
# 絶対誤差
# ==========================================

df["absolute_error"] = df["error_kmh"].abs()


# ==========================================
# ① 誤差が大きい動画
# ==========================================

print("\n")
print("=" * 60)
print("① 予測誤差が大きい動画 TOP10")
print("=" * 60)

cols = [
    "video",
    "actual_speed_kmh",
    "predicted_speed_kmh",
    "error_kmh",
    "absolute_error",
    "num_frames",
    "average_pixel_speed",
    "max_pixel_speed",
    "path_straightness",
    "mean_confidence",
    "min_confidence",
]

top_error = df[cols].sort_values("absolute_error", ascending=False).head(10)

print(top_error.to_string(index=False))


# ==========================================
# ② 誤差が小さい動画
# ==========================================

print("\n")
print("=" * 60)
print("② 予測誤差が小さい動画 TOP10")
print("=" * 60)

low_error = df[cols].sort_values("absolute_error", ascending=True).head(10)

print(low_error.to_string(index=False))


# ==========================================
# ③ 各特徴量と絶対誤差の相関
# ==========================================

print("\n")
print("=" * 60)
print("③ 各特徴量と予測誤差の相関")
print("=" * 60)

feature_cols = [
    "num_frames",
    "duration_sec",
    "total_distance_px",
    "start_end_distance_px",
    "average_pixel_speed",
    "max_pixel_speed",
    "total_dx",
    "total_dy",
    "direction_change",
    "mean_step_distance",
    "std_step_distance",
    "max_step_distance",
    "median_step_distance",
    "path_straightness",
    "trajectory_angle_deg",
    "mean_dx",
    "mean_dy",
    "std_dx",
    "std_dy",
    "std_pixel_speed",
    "mean_confidence",
    "min_confidence",
]

correlations = []

for feature in feature_cols:
    if feature in df.columns:
        corr = df[feature].corr(df["absolute_error"])
        correlations.append((feature, corr))

correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

for feature, corr in correlations:
    print(f"{feature:30s} : {corr:+.3f}")


# ==========================================
# ④ フレーム数ごとの平均誤差
# ==========================================

print("\n")
print("=" * 60)
print("④ 追跡フレーム数ごとの平均誤差")
print("=" * 60)

frame_summary = (
    df.groupby("num_frames")
    .agg(
        videos=("video", "count"),
        mean_error=("absolute_error", "mean"),
        max_error=("absolute_error", "max"),
    )
    .sort_index()
)

print(frame_summary.to_string())


# ==========================================
# ⑤ 信頼度ごとの平均誤差
# ==========================================

print("\n")
print("=" * 60)
print("⑤ 平均検出信頼度ごとの平均誤差")
print("=" * 60)

df["confidence_group"] = pd.cut(
    df["mean_confidence"],
    bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
    labels=[
        "0.0-0.3",
        "0.3-0.5",
        "0.5-0.7",
        "0.7-0.9",
        "0.9-1.0",
    ],
)

confidence_summary = df.groupby("confidence_group", observed=False).agg(
    videos=("video", "count"),
    mean_error=("absolute_error", "mean"),
    max_error=("absolute_error", "max"),
)

print(confidence_summary.to_string())


# ==========================================
# ⑥ 軌跡直線性ごとの平均誤差
# ==========================================

print("\n")
print("=" * 60)
print("⑥ 軌跡直線性ごとの平均誤差")
print("=" * 60)

df["straightness_group"] = pd.cut(
    df["path_straightness"],
    bins=[0, 0.8, 0.9, 0.95, 0.98, 1.0],
    labels=[
        "0.0-0.8",
        "0.8-0.9",
        "0.9-0.95",
        "0.95-0.98",
        "0.98-1.0",
    ],
)

straightness_summary = df.groupby("straightness_group", observed=False).agg(
    videos=("video", "count"),
    mean_error=("absolute_error", "mean"),
    max_error=("absolute_error", "max"),
)

print(straightness_summary.to_string())


# ==========================================
# ⑦ 実測球速とピクセル速度
# ==========================================

print("\n")
print("=" * 60)
print("⑦ 実際の球速とピクセル速度")
print("=" * 60)

speed_cols = [
    "video",
    "actual_speed_kmh",
    "average_pixel_speed",
    "max_pixel_speed",
    "num_frames",
    "path_straightness",
]

speed_table = df[speed_cols].sort_values("actual_speed_kmh")

print(speed_table.to_string(index=False))


# ==========================================
# ⑧ 保存
# ==========================================

output_file = BASE_DIR / "output" / "ball_model_diagnostic.csv"

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n")
print("=" * 60)
print("診断完了")
print("=" * 60)

print("診断結果を保存しました:")
print(output_file)
