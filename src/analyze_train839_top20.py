from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance

# ============================================================
# 設定
# ============================================================

INPUT_CSV = Path("output/ball_tracking_dataset_train11_839.csv")

MODEL_PATH = Path("output/train839_feature_selection/" "best_train839_speed_model.pkl")

OUTPUT_DIR = Path("output/train839_top20_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "speed_kmh"


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
# 表示
# ============================================================

print("=" * 60)
print("839本 TOP_20 球速推定モデル詳細解析")
print("=" * 60)

print()
print("入力CSV:")
print(INPUT_CSV)

print()
print("モデル:")
print(MODEL_PATH)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(INPUT_CSV)

print()
print("読み込みデータ数:", len(df))


# ============================================================
# 必要列確認
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:

    print()
    print("エラー")
    print("必要な列がありません:")

    for column in missing_columns:
        print(" -", column)

    raise ValueError("必要な列がCSVにありません")


# ============================================================
# データ準備
# ============================================================

work_df = df[FEATURES + [TARGET]].copy()

work_df = work_df.replace([np.inf, -np.inf], np.nan)

before = len(work_df)

work_df = work_df.dropna(subset=[TARGET])

X = work_df[FEATURES].copy()

y = work_df[TARGET].copy()

X = X.fillna(X.median(numeric_only=True))

print()
print("=" * 60)
print("データ確認")
print("=" * 60)

print()
print("処理前:", before)
print("解析データ:", len(work_df))
print("除外:", before - len(work_df))

print()
print("特徴量数:", len(FEATURES))


# ============================================================
# LOOCV
# ============================================================

print()
print("=" * 60)
print("LOOCV評価")
print("=" * 60)

print()
print("LOOCV実行中...")


model_params = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
}


loo = LeaveOneOut()

predictions = np.zeros(len(y), dtype=float)


for train_index, test_index in loo.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]

    model = RandomForestRegressor(**model_params)

    model.fit(X_train, y_train)

    predictions[test_index[0]] = model.predict(X_test)[0]


# ============================================================
# 全体評価
# ============================================================

mae = mean_absolute_error(y, predictions)

rmse = np.sqrt(mean_squared_error(y, predictions))

r2 = r2_score(y, predictions)

print()
print("MAE  :", f"{mae:.3f}", "km/h")
print("RMSE :", f"{rmse:.3f}", "km/h")
print("R²   :", f"{r2:.3f}")


# ============================================================
# 予測結果
# ============================================================

result = df.loc[work_df.index].copy()

result["actual_speed_kmh"] = y.values

result["predicted_speed_kmh"] = predictions

result["error_kmh"] = result["predicted_speed_kmh"] - result["actual_speed_kmh"]

result["absolute_error_kmh"] = result["error_kmh"].abs()


# ============================================================
# 誤差統計
# ============================================================

print()
print("=" * 60)
print("誤差統計")
print("=" * 60)

print()
print("平均絶対誤差:", f"{result['absolute_error_kmh'].mean():.3f}", "km/h")

print("中央値絶対誤差:", f"{result['absolute_error_kmh'].median():.3f}", "km/h")

print("最大絶対誤差:", f"{result['absolute_error_kmh'].max():.3f}", "km/h")

print("平均誤差:", f"{result['error_kmh'].mean():.3f}", "km/h")


# ============================================================
# 誤差区間
# ============================================================

print()
print("=" * 60)
print("誤差区間")
print("=" * 60)

error_bins = [0, 2, 5, 10, 15, np.inf]

error_labels = [
    "0-2 km/h",
    "2-5 km/h",
    "5-10 km/h",
    "10-15 km/h",
    "15+ km/h",
]

result["error_range"] = pd.cut(
    result["absolute_error_kmh"], bins=error_bins, labels=error_labels, right=False
)

error_range = (
    result["error_range"].value_counts().reindex(error_labels).fillna(0).astype(int)
)

error_range_percent = error_range / len(result) * 100

for label in error_labels:

    count = error_range[label]

    percent = error_range_percent[label]

    print(f"{label:12s}: " f"{count:4d}本 " f"({percent:.1f}%)")


error_range_df = pd.DataFrame(
    {
        "error_range": error_labels,
        "count": [error_range[label] for label in error_labels],
        "percentage": [error_range_percent[label] for label in error_labels],
    }
)


# ============================================================
# 球速帯
# ============================================================

print()
print("=" * 60)
print("球速帯ごとの誤差")
print("=" * 60)


def speed_band(speed):

    if speed < 140:
        return "130-139"

    elif speed < 150:
        return "140-149"

    elif speed < 160:
        return "150-159"

    else:
        return "160-169"


result["speed_band"] = result["actual_speed_kmh"].apply(speed_band)


speed_order = [
    "130-139",
    "140-149",
    "150-159",
    "160-169",
]


speed_group_rows = []


for band in speed_order:

    group = result[result["speed_band"] == band]

    if len(group) == 0:
        continue

    group_mae = mean_absolute_error(
        group["actual_speed_kmh"], group["predicted_speed_kmh"]
    )

    group_rmse = np.sqrt(
        mean_squared_error(group["actual_speed_kmh"], group["predicted_speed_kmh"])
    )

    group_bias = group["error_kmh"].mean()

    print()
    print(band)

    print("動画数:", len(group))

    print("MAE :", f"{group_mae:.3f}", "km/h")

    print("RMSE:", f"{group_rmse:.3f}", "km/h")

    print("平均誤差:", f"{group_bias:.3f}", "km/h")

    print("実測平均:", f"{group['actual_speed_kmh'].mean():.3f}")

    print("予測平均:", f"{group['predicted_speed_kmh'].mean():.3f}")

    speed_group_rows.append(
        {
            "speed_band": band,
            "videos": len(group),
            "MAE_kmh": group_mae,
            "RMSE_kmh": group_rmse,
            "mean_error_kmh": group_bias,
            "actual_mean_kmh": group["actual_speed_kmh"].mean(),
            "predicted_mean_kmh": group["predicted_speed_kmh"].mean(),
        }
    )


speed_group_df = pd.DataFrame(speed_group_rows)


# ============================================================
# 最大誤差ランキング
# ============================================================

print()
print("=" * 60)
print("最大誤差ランキング TOP20")
print("=" * 60)


worst_20 = result.sort_values("absolute_error_kmh", ascending=False).head(20)


display_columns = [
    "video",
    "actual_speed_kmh",
    "predicted_speed_kmh",
    "error_kmh",
    "absolute_error_kmh",
]


available_display_columns = [
    column for column in display_columns if column in worst_20.columns
]


print()

print(worst_20[available_display_columns].to_string(index=False))


# ============================================================
# 予測値分布
# ============================================================

print()
print("=" * 60)
print("予測値分布")
print("=" * 60)

print()
print("実測最低:", f"{y.min():.3f}", "km/h")

print("実測最高:", f"{y.max():.3f}", "km/h")

print("予測最低:", f"{predictions.min():.3f}", "km/h")

print("予測最高:", f"{predictions.max():.3f}", "km/h")

print("実測平均:", f"{y.mean():.3f}", "km/h")

print("予測平均:", f"{predictions.mean():.3f}", "km/h")


# ============================================================
# 保存
# ============================================================

prediction_output = OUTPUT_DIR / "predictions_with_analysis.csv"

result.to_csv(prediction_output, index=False, encoding="utf-8-sig")


error_range_output = OUTPUT_DIR / "error_ranges.csv"

error_range_df.to_csv(error_range_output, index=False, encoding="utf-8-sig")


speed_group_output = OUTPUT_DIR / "speed_group_performance.csv"

speed_group_df.to_csv(speed_group_output, index=False, encoding="utf-8-sig")


worst_output = OUTPUT_DIR / "worst_20_predictions.csv"

worst_20[available_display_columns].to_csv(
    worst_output, index=False, encoding="utf-8-sig"
)


# ============================================================
# 全データモデルの特徴量重要度
# ============================================================

print()
print("=" * 60)
print("特徴量重要度")
print("=" * 60)

print()
print("最終モデル読み込み:")

print(MODEL_PATH)


final_model = joblib.load(MODEL_PATH)


print()
print("Permutation Importance計算中...")


perm = permutation_importance(
    final_model,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
)


importance_df = importance_df.sort_values(
    "importance_mean", ascending=False
).reset_index(drop=True)


print()

print(importance_df.to_string(index=False))


importance_output = OUTPUT_DIR / "feature_importance.csv"

importance_df.to_csv(importance_output, index=False, encoding="utf-8-sig")


# ============================================================
# start_x依存チェック
# ============================================================

print()
print("=" * 60)
print("start_x依存チェック")
print("=" * 60)

print()
print("start_xを含むTOP_20モデルが" "良い結果になっています。")

print("そのため、start_xを除外した場合の" "性能も確認します。")


features_without_start_x = [feature for feature in FEATURES if feature != "start_x"]


X_no_start_x = X[features_without_start_x].copy()


print()
print("start_x除外後の特徴量数:", len(features_without_start_x))


loo_no_start_x = LeaveOneOut()

predictions_no_start_x = np.zeros(len(y), dtype=float)


for train_index, test_index in loo_no_start_x.split(X_no_start_x):

    X_train = X_no_start_x.iloc[train_index]

    X_test = X_no_start_x.iloc[test_index]

    y_train = y.iloc[train_index]

    model = RandomForestRegressor(**model_params)

    model.fit(X_train, y_train)

    predictions_no_start_x[test_index[0]] = model.predict(X_test)[0]


no_start_x_mae = mean_absolute_error(y, predictions_no_start_x)

no_start_x_rmse = np.sqrt(mean_squared_error(y, predictions_no_start_x))

no_start_x_r2 = r2_score(y, predictions_no_start_x)


print()
print("start_x除外モデル")

print("MAE :", f"{no_start_x_mae:.3f}", "km/h")

print("RMSE:", f"{no_start_x_rmse:.3f}", "km/h")

print("R²  :", f"{no_start_x_r2:.3f}")


# ============================================================
# 比較結果保存
# ============================================================

start_x_comparison = pd.DataFrame(
    [
        {
            "model": "TOP20_with_start_x",
            "features": 20,
            "MAE_kmh": mae,
            "RMSE_kmh": rmse,
            "R2": r2,
        },
        {
            "model": "TOP20_without_start_x",
            "features": len(features_without_start_x),
            "MAE_kmh": no_start_x_mae,
            "RMSE_kmh": no_start_x_rmse,
            "R2": no_start_x_r2,
        },
    ]
)


start_x_output = OUTPUT_DIR / "start_x_comparison.csv"

start_x_comparison.to_csv(start_x_output, index=False, encoding="utf-8-sig")


# ============================================================
# サマリー
# ============================================================

summary_path = OUTPUT_DIR / "analysis_summary.txt"


with open(summary_path, "w", encoding="utf-8") as f:

    f.write("839本 TOP_20 球速推定モデル解析\n")

    f.write("=" * 60 + "\n\n")

    f.write(f"動画数: {len(y)}\n")

    f.write(f"特徴量数: {len(FEATURES)}\n\n")

    f.write("LOOCV結果\n")

    f.write(f"MAE: {mae:.6f} km/h\n")

    f.write(f"RMSE: {rmse:.6f} km/h\n")

    f.write(f"R2: {r2:.6f}\n\n")

    f.write("start_x除外結果\n")

    f.write(f"MAE: {no_start_x_mae:.6f} km/h\n")

    f.write(f"RMSE: {no_start_x_rmse:.6f} km/h\n")

    f.write(f"R2: {no_start_x_r2:.6f}\n\n")

    f.write("使用特徴量\n")

    for feature in FEATURES:

        f.write(f"- {feature}\n")


# ============================================================
# 完了
# ============================================================

print()
print("=" * 60)
print("解析完了")
print("=" * 60)

print()
print("保存先:")
print(OUTPUT_DIR)

print()
print("作成ファイル:")

print(" - predictions_with_analysis.csv")

print(" - error_ranges.csv")

print(" - speed_group_performance.csv")

print(" - worst_20_predictions.csv")

print(" - feature_importance.csv")

print(" - start_x_comparison.csv")

print(" - analysis_summary.txt")

print()
print("=" * 60)
print("完了")
print("=" * 60)
