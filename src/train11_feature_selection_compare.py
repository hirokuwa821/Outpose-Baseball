import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

TARGET = "speed_kmh"


# ==========================================
# Permutation Importanceの結果から
# 使用する特徴量を設定
# ==========================================

FEATURES_ALL = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
    "max_step_distance",
    "std_step_distance",
    "trajectory_angle_deg",
    "max_pixel_speed",
    "average_pixel_speed",
    "median_step_distance",
    "total_dy",
    "total_distance_px",
    "mean_dy",
    "start_end_distance_px",
    "total_dx",
    "mean_step_distance",
    "duration_sec",
    "mean_dx",
    "num_frames",
    "direction_change",
]


# ==========================================
# 特徴量セット
# ==========================================

feature_sets = {
    "ALL_20": FEATURES_ALL,
    "TOP_10": FEATURES_ALL[:10],
    "TOP_7": FEATURES_ALL[:7],
    "TOP_4": FEATURES_ALL[:4],
}


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

df = df[df["track_success"] == 1].copy()


# ==========================================
# 基本情報
# ==========================================

print("=" * 60)
print("train-11 特徴量選択比較")
print("=" * 60)

print()

print("データ数:", len(df))

print()

print("球速範囲")

print(f"最低: {df[TARGET].min():.3f} km/h")
print(f"最高: {df[TARGET].max():.3f} km/h")
print(f"平均: {df[TARGET].mean():.3f} km/h")


# ==========================================
# Leave-One-Out
# ==========================================

cv = LeaveOneOut()


# ==========================================
# 結果
# ==========================================

results = []


# ==========================================
# 各特徴量セットを比較
# ==========================================

for set_name, features in feature_sets.items():

    print()
    print("=" * 60)
    print(set_name)
    print("=" * 60)

    print()

    print("使用特徴量数:", len(features))

    print()

    print("特徴量:")

    for feature in features:

        print(" -", feature)

    # --------------------------------------
    # 欠損値除外
    # --------------------------------------

    data = df.dropna(subset=features + [TARGET]).copy()

    X = data[features]

    y = data[TARGET]

    # --------------------------------------
    # RandomForest
    # --------------------------------------

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    # --------------------------------------
    # Leave-One-Out予測
    # --------------------------------------

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    # --------------------------------------
    # 評価
    # --------------------------------------

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    # --------------------------------------
    # 表示
    # --------------------------------------

    print()

    print(f"MAE  : {mae:.3f} km/h")

    print(f"RMSE : {rmse:.3f} km/h")

    print(f"R2   : {r2:.3f}")

    # --------------------------------------
    # 保存
    # --------------------------------------

    prediction_df = pd.DataFrame(
        {
            "video": data["video"].values,
            "actual_speed_kmh": y.values,
            "predicted_speed_kmh": predictions,
        }
    )

    prediction_df["error_kmh"] = (
        prediction_df["predicted_speed_kmh"] - prediction_df["actual_speed_kmh"]
    )

    prediction_df["absolute_error_kmh"] = prediction_df["error_kmh"].abs()

    output_path = f"output/" f"randomforest_predictions_{set_name}.csv"

    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("予測結果保存:")

    print(output_path)

    # --------------------------------------
    # 比較用
    # --------------------------------------

    results.append(
        {
            "feature_set": set_name,
            "features": len(features),
            "videos": len(data),
            "MAE_kmh": mae,
            "RMSE_kmh": rmse,
            "R2": r2,
        }
    )


# ==========================================
# 比較結果
# ==========================================

results_df = pd.DataFrame(results)


# MAEが小さい順

results_df = results_df.sort_values("MAE_kmh", ascending=True).reset_index(drop=True)


# ==========================================
# 保存
# ==========================================

OUTPUT_CSV = "output/" "train11_feature_selection_comparison.csv"


results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================
# 最終表示
# ==========================================

print()

print("=" * 60)

print("特徴量セット比較結果")

print("=" * 60)

print()

print(results_df.to_string(index=False))

print()

print("保存先:")

print(OUTPUT_CSV)
