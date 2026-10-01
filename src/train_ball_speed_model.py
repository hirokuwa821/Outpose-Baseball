import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_40.csv"


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


# ------------------------------------------
# 追跡成功した動画のみ使用
# ------------------------------------------

df = df[df["track_success"] == 1].copy()


# ------------------------------------------
# 欠損値を除外
# ------------------------------------------

df = df.dropna(subset=FEATURES + [TARGET])


X = df[FEATURES]
y = df[TARGET]


# ==========================================
# 基本情報表示
# ==========================================

print("=" * 60)
print("球速予測モデル")
print("=" * 60)

print("使用動画数:", len(df))

print("特徴量数:", len(FEATURES))

print()


# ==========================================
# Leave-One-Out交差検証
# ==========================================

cv = LeaveOneOut()


# ==========================================
# モデル設定
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
# モデル比較
# ==========================================

for model_name, model in models.items():

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    # --------------------------------------
    # Leave-One-Outで予測
    # --------------------------------------

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    # --------------------------------------
    # 評価指標
    # --------------------------------------

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    print(f"MAE  : {mae:.3f} km/h")

    print(f"RMSE : {rmse:.3f} km/h")

    print(f"R2   : {r2:.3f}")

    # --------------------------------------
    # モデル比較用に保存
    # --------------------------------------

    results.append(
        {
            "model": model_name,
            "MAE_kmh": mae,
            "RMSE_kmh": rmse,
            "R2": r2,
        }
    )

    # ======================================
    # 各動画の予測結果
    # ======================================

    prediction_df = pd.DataFrame(
        {
            "video": df["video"].values,
            "actual_speed_kmh": y.values,
            "predicted_speed_kmh": predictions,
        }
    )

    # --------------------------------------
    # 誤差
    # --------------------------------------

    prediction_df["error_kmh"] = (
        prediction_df["predicted_speed_kmh"] - prediction_df["actual_speed_kmh"]
    )

    prediction_df["absolute_error_kmh"] = prediction_df["error_kmh"].abs()

    # --------------------------------------
    # 予測結果を保存
    # --------------------------------------

    output_path = f"output/" f"{model_name.lower()}_" f"ball_speed_predictions_40.csv"

    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("予測結果保存:")

    print(output_path)


# ==========================================
# モデル比較結果を保存
# ==========================================

results_df = pd.DataFrame(results)


comparison_output_path = "output/ball_speed_model_comparison_40.csv"


results_df.to_csv(
    comparison_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================
# 結果表示
# ==========================================

print()

print("=" * 60)
print("モデル比較")
print("=" * 60)

print(results_df.to_string(index=False))

print()

print("比較結果保存:")

print(comparison_output_path)
