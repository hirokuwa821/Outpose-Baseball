import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_CSV = "output/train11_feature_importance.csv"

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
    "mean_confidence",
    "min_confidence",
]

TARGET = "speed_kmh"


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("=" * 60)
print("train-11 特徴量重要度解析")
print("=" * 60)
print()

print("読み込みデータ数:", len(df))

# 必要な列だけ使用
df = df.dropna(subset=FEATURES + [TARGET]).copy()

X = df[FEATURES]
y = df[TARGET]


# ==========================================
# ExtraTrees
# ==========================================

model = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    max_features=0.8,
    min_samples_leaf=2,
    n_jobs=-1,
)

model.fit(X, y)


# ==========================================
# Gini系の特徴量重要度
# ==========================================

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }
)

importance_df = importance_df.sort_values("importance", ascending=False).reset_index(
    drop=True
)


# ==========================================
# 累積重要度
# ==========================================

importance_df["cumulative_importance"] = importance_df["importance"].cumsum()


# ==========================================
# Permutation Importance
# ==========================================

print()
print("=" * 60)
print("Permutation Importance")
print("=" * 60)

perm = permutation_importance(
    model,
    X,
    y,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
)

perm_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }
)

perm_df = perm_df.sort_values(
    "permutation_importance_mean", ascending=False
).reset_index(drop=True)


# ==========================================
# 2つを結合
# ==========================================

result_df = importance_df.merge(perm_df, on="feature", how="left")


# ==========================================
# 保存
# ==========================================

Path("output").mkdir(parents=True, exist_ok=True)

result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


# ==========================================
# 表示
# ==========================================

print()
print("=" * 60)
print("特徴量重要度")
print("=" * 60)
print()

print(
    result_df[
        [
            "feature",
            "importance",
            "cumulative_importance",
            "permutation_importance_mean",
            "permutation_importance_std",
        ]
    ].to_string(index=False)
)

print()
print("=" * 60)
print("保存完了")
print("=" * 60)
print()
print("保存先:", OUTPUT_CSV)
