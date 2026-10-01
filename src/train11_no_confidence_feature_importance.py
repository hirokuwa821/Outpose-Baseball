import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_CSV = "output/train11_no_confidence_feature_importance.csv"

TARGET = "speed_kmh"


# ==========================================
# confidence系を除いた20特徴量
# ==========================================

FEATURES = [
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
]


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

df = df[df["track_success"] == 1].copy()

df = df.dropna(subset=FEATURES + [TARGET])


X = df[FEATURES]

y = df[TARGET]


# ==========================================
# 基本情報
# ==========================================

print("=" * 60)
print("train-11 confidence除外版 特徴量重要度解析")
print("=" * 60)

print()

print("データ数:", len(df))

print("特徴量数:", len(FEATURES))

print()


# ==========================================
# RandomForest
# ==========================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_features=0.8,
    min_samples_leaf=2,
    n_jobs=-1,
)


# ==========================================
# 全データで学習
# ==========================================

model.fit(X, y)


# ==========================================
# Permutation Importance
# ==========================================

print("=" * 60)
print("Permutation Importance")
print("=" * 60)

importance = permutation_importance(
    model,
    X,
    y,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)


# ==========================================
# 結果整理
# ==========================================

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "permutation_importance_mean": importance.importances_mean,
        "permutation_importance_std": importance.importances_std,
    }
)


# ==========================================
# 絶対値ではなく順位付け
# ==========================================

importance_df = importance_df.sort_values(
    "permutation_importance_mean", ascending=False
).reset_index(drop=True)


# ==========================================
# 正規化重要度
# ==========================================

total = importance_df["permutation_importance_mean"].sum()


if total != 0:

    importance_df["importance"] = importance_df["permutation_importance_mean"] / total

else:

    importance_df["importance"] = 0


# ==========================================
# 累積重要度
# ==========================================

importance_df["cumulative_importance"] = importance_df["importance"].cumsum()


# ==========================================
# 表示
# ==========================================

print()

print("=" * 60)
print("特徴量重要度")
print("=" * 60)

print()

print(
    importance_df[
        [
            "feature",
            "importance",
            "cumulative_importance",
            "permutation_importance_mean",
            "permutation_importance_std",
        ]
    ].to_string(index=False)
)


# ==========================================
# 保存
# ==========================================

importance_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


print()

print("=" * 60)
print("保存完了")
print("=" * 60)

print()

print("保存先:")

print(OUTPUT_CSV)
