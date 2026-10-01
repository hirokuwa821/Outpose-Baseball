import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

TARGET = "speed_kmh"


# ==========================================
# 信頼度・追跡品質系を除いた特徴量
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
print("train-11 特徴量絞り込みモデル")
print("=" * 60)

print()
print("データ数:", len(df))
print("特徴量数:", len(FEATURES))

print()
print("球速範囲")
print(f"最低: {y.min():.3f} km/h")
print(f"最高: {y.max():.3f} km/h")
print(f"平均: {y.mean():.3f} km/h")


# ==========================================
# Leave-One-Out
# ==========================================

cv = LeaveOneOut()


# ==========================================
# モデル
# ==========================================

models = {
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
}


results = []


# ==========================================
# 学習・評価
# ==========================================

for model_name, model in models.items():

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    print()
    print(f"MAE  : {mae:.3f} km/h")
    print(f"RMSE : {rmse:.3f} km/h")
    print(f"R2   : {r2:.3f}")

    # ======================================
    # 予測結果
    # ======================================

    prediction_df = pd.DataFrame(
        {
            "video": df["video"].values,
            "actual_speed_kmh": y.values,
            "predicted_speed_kmh": predictions,
        }
    )

    prediction_df["error_kmh"] = (
        prediction_df["predicted_speed_kmh"] - prediction_df["actual_speed_kmh"]
    )

    prediction_df["absolute_error_kmh"] = prediction_df["error_kmh"].abs()

    output_path = (
        f"output/" f"{model_name.lower()}_" f"ball_speed_predictions_no_confidence.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("予測結果保存:")
    print(output_path)

    results.append(
        {
            "model": model_name,
            "videos": len(df),
            "features": len(FEATURES),
            "MAE_kmh": mae,
            "RMSE_kmh": rmse,
            "R2": r2,
        }
    )


# ==========================================
# 比較結果
# ==========================================

results_df = pd.DataFrame(results)

comparison_output = "output/" "ball_speed_model_comparison_no_confidence.csv"

results_df.to_csv(
    comparison_output,
    index=False,
    encoding="utf-8-sig",
)


print()
print("=" * 60)
print("モデル比較")
print("=" * 60)

print(results_df.to_string(index=False))

print()
print("保存先:")
print(comparison_output)
