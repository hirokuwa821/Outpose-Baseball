import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 設定
# ============================================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_COMPARISON = "output/ball_speed_model_comparison_train11.csv"

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


# ============================================================
# データ読み込み
# ============================================================

print("=" * 60)
print("train-11 球速推定モデル")
print("=" * 60)
print()

df = pd.read_csv(INPUT_CSV)

print("読み込みデータ数:", len(df))
print()


# ============================================================
# 必要な列の確認
# ============================================================

print("特徴量の確認")
print("-" * 60)

missing_features = [feature for feature in FEATURES if feature not in df.columns]

if missing_features:
    print("不足している特徴量:")
    for feature in missing_features:
        print("  ", feature)

    print()
    print("CSVに存在する列:")
    print(df.columns.tolist())

    raise SystemExit("必要な特徴量がありません。")


if TARGET not in df.columns:
    raise SystemExit(f"正解値 {TARGET} がCSVにありません。")


print("特徴量:", len(FEATURES))
print("正解値:", TARGET)
print()


# ============================================================
# 欠損値除去
# ============================================================

before = len(df)

df = df.dropna(subset=FEATURES + [TARGET]).copy()

after = len(df)

print("欠損値処理")
print("処理前:", before)
print("処理後:", after)
print("除外:", before - after)
print()


# ============================================================
# 学習データ
# ============================================================

X = df[FEATURES]
y = df[TARGET]


print("=" * 60)
print("学習データ")
print("=" * 60)

print("動画数:", len(df))
print("特徴量数:", len(FEATURES))
print()

print("球速範囲:")
print(f"最低: {y.min():.3f} km/h")
print(f"最高: {y.max():.3f} km/h")
print(f"平均: {y.mean():.3f} km/h")
print()


# ============================================================
# モデル
# ============================================================

models = {
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
}


# ============================================================
# Leave-One-Out交差検証
# ============================================================

cv = LeaveOneOut()

results = []


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

    # --------------------------------------------------------
    # 評価
    # --------------------------------------------------------

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    print()
    print(f"MAE  : {mae:.3f} km/h")
    print(f"RMSE : {rmse:.3f} km/h")
    print(f"R²   : {r2:.3f}")
    print()

    # --------------------------------------------------------
    # 結果保存用データ
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 追跡品質情報があれば追加
    # --------------------------------------------------------

    quality_columns = [
        "path_straightness",
        "mean_confidence",
        "min_confidence",
        "num_frames",
        "duration_sec",
    ]

    for column in quality_columns:

        if column in df.columns:
            prediction_df[column] = df[column].values

    # --------------------------------------------------------
    # 個別予測結果保存
    # --------------------------------------------------------

    output_path = (
        f"output/" f"{model_name.lower()}_" f"ball_speed_predictions_train11.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("予測結果保存:")
    print(output_path)

    # --------------------------------------------------------
    # モデル比較結果
    # --------------------------------------------------------

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


# ============================================================
# モデル比較結果
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_COMPARISON,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 結果表示
# ============================================================

print()
print("=" * 60)
print("モデル比較結果")
print("=" * 60)
print()

print(results_df.to_string(index=False))

print()
print("保存先:")
print(OUTPUT_COMPARISON)


# ============================================================
# 各動画の誤差確認
# ============================================================

print()
print("=" * 60)
print("各動画の予測誤差")
print("=" * 60)
print()

# ExtraTreesの結果を読み込む
extra_path = "output/" "extratrees_ball_speed_predictions_train11.csv"

extra_df = pd.read_csv(extra_path)

extra_df = extra_df.sort_values("absolute_error_kmh", ascending=False)

print(
    extra_df[
        [
            "video",
            "actual_speed_kmh",
            "predicted_speed_kmh",
            "error_kmh",
            "absolute_error_kmh",
        ]
    ].to_string(index=False)
)

print()
print("以上")
