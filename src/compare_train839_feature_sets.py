from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

# ============================================================
# 設定
# ============================================================

INPUT_CSV = Path("output/ball_tracking_dataset_train11_839.csv")

OUTPUT_DIR = Path("output/train839_feature_selection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 表示
# ============================================================

print("=" * 60)
print("839本 特徴量セット比較")
print("=" * 60)

print()
print("入力CSV:")
print(INPUT_CSV)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(INPUT_CSV)

print()
print("読み込みデータ数:", len(df))


# ============================================================
# 正解値
# ============================================================

TARGET = "speed_kmh"

if TARGET not in df.columns:
    raise ValueError(f"正解値 {TARGET} がCSVにありません")


# ============================================================
# 使用候補特徴量
# ============================================================
#
# train-11で作成した特徴量30個のうち、
# 球速推定に使える数値特徴量を対象にする。
#
# speed_kmh自体は絶対に特徴量へ入れない。
#

EXCLUDE_COLUMNS = {
    TARGET,
    "video",
    "video_name",
    "file",
    "filename",
    "path",
    "speed_mph",
    "hand",
}


numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()


candidate_features = [
    column for column in numeric_columns if column not in EXCLUDE_COLUMNS
]


print()
print("=" * 60)
print("候補特徴量")
print("=" * 60)

print()
print("特徴量数:", len(candidate_features))

for feature in candidate_features:
    print(" -", feature)


# ============================================================
# 欠損値処理
# ============================================================

print()
print("=" * 60)
print("欠損値処理")
print("=" * 60)

before = len(df)

work_df = df[candidate_features + [TARGET]].copy()

work_df = work_df.replace([np.inf, -np.inf], np.nan)

work_df = work_df.dropna(subset=[TARGET])

print()
print("処理前:", before)
print("処理後:", len(work_df))
print("除外:", before - len(work_df))


# ============================================================
# 特徴量の欠損値を中央値で補完
# ============================================================

X_all = work_df[candidate_features].copy()
y = work_df[TARGET].copy()

X_all = X_all.fillna(X_all.median(numeric_only=True))


# ============================================================
# まず全特徴量でPermutation Importance
# ============================================================

print()
print("=" * 60)
print("特徴量重要度解析")
print("=" * 60)

print()
print("全特徴量でRandomForestを学習...")


importance_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
)

importance_model.fit(X_all, y)


print("Permutation Importance計算中...")


importance_result = permutation_importance(
    importance_model,
    X_all,
    y,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)


importance_df = pd.DataFrame(
    {
        "feature": candidate_features,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }
)


importance_df = importance_df.sort_values(
    "importance_mean", ascending=False
).reset_index(drop=True)


importance_df["rank"] = np.arange(len(importance_df)) + 1


print()
print("特徴量重要度 TOP30")
print("-" * 60)

print(
    importance_df[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ].to_string(index=False)
)


importance_path = OUTPUT_DIR / "feature_importance_839.csv"

importance_df.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 特徴量セット作成
# ============================================================

ranked_features = importance_df["feature"].tolist()


feature_set_sizes = [
    4,
    7,
    10,
    15,
    20,
    len(ranked_features),
]


# 重複除去
feature_set_sizes = list(dict.fromkeys(feature_set_sizes))

feature_sets = {}

for size in feature_set_sizes:

    if size <= len(ranked_features):

        feature_sets[f"TOP_{size}"] = ranked_features[:size]


# ============================================================
# LOOCV関数
# ============================================================


def evaluate_model(
    model_name,
    model,
    X,
    y,
):
    """
    Leave-One-Out Cross Validation
    """

    loo = LeaveOneOut()

    predictions = np.zeros(len(y), dtype=float)

    print()
    print(f"{model_name} LOOCV実行中...")

    for train_index, test_index in loo.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]

        model.fit(X_train, y_train)

        predictions[test_index[0]] = model.predict(X_test)[0]

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))

    r2 = r2_score(y, predictions)

    return (
        mae,
        rmse,
        r2,
        predictions,
    )


# ============================================================
# モデル比較
# ============================================================

results = []

prediction_files = []


for feature_set_name, features in feature_sets.items():

    print()
    print("=" * 60)
    print(feature_set_name)
    print("=" * 60)

    print()
    print("使用特徴量数:", len(features))

    print()
    print("特徴量:")

    for feature in features:
        print(" -", feature)

    X = X_all[features].copy()

    # --------------------------------------------------------
    # RandomForest
    # --------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

    (
        rf_mae,
        rf_rmse,
        rf_r2,
        rf_predictions,
    ) = evaluate_model(
        "RandomForest",
        rf_model,
        X,
        y,
    )

    print()
    print("RandomForest")
    print(f"MAE  : {rf_mae:.3f} km/h")
    print(f"RMSE : {rf_rmse:.3f} km/h")
    print(f"R²   : {rf_r2:.3f}")

    rf_prediction_df = pd.DataFrame(
        {
            "actual_speed_kmh": y.values,
            "predicted_speed_kmh": rf_predictions,
        }
    )

    rf_prediction_df["error_kmh"] = (
        rf_prediction_df["predicted_speed_kmh"] - rf_prediction_df["actual_speed_kmh"]
    )

    rf_prediction_df["absolute_error_kmh"] = rf_prediction_df["error_kmh"].abs()

    rf_output = OUTPUT_DIR / f"randomforest_predictions_{feature_set_name}.csv"

    rf_prediction_df.to_csv(
        rf_output,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_files.append(rf_output)

    results.append(
        {
            "feature_set": feature_set_name,
            "model": "RandomForest",
            "features": len(features),
            "videos": len(y),
            "MAE_kmh": rf_mae,
            "RMSE_kmh": rf_rmse,
            "R2": rf_r2,
        }
    )

    # --------------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------------

    et_model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

    (
        et_mae,
        et_rmse,
        et_r2,
        et_predictions,
    ) = evaluate_model(
        "ExtraTrees",
        et_model,
        X,
        y,
    )

    print()
    print("ExtraTrees")
    print(f"MAE  : {et_mae:.3f} km/h")
    print(f"RMSE : {et_rmse:.3f} km/h")
    print(f"R²   : {et_r2:.3f}")

    et_prediction_df = pd.DataFrame(
        {
            "actual_speed_kmh": y.values,
            "predicted_speed_kmh": et_predictions,
        }
    )

    et_prediction_df["error_kmh"] = (
        et_prediction_df["predicted_speed_kmh"] - et_prediction_df["actual_speed_kmh"]
    )

    et_prediction_df["absolute_error_kmh"] = et_prediction_df["error_kmh"].abs()

    et_output = OUTPUT_DIR / f"extratrees_predictions_{feature_set_name}.csv"

    et_prediction_df.to_csv(
        et_output,
        index=False,
        encoding="utf-8-sig",
    )

    prediction_files.append(et_output)

    results.append(
        {
            "feature_set": feature_set_name,
            "model": "ExtraTrees",
            "features": len(features),
            "videos": len(y),
            "MAE_kmh": et_mae,
            "RMSE_kmh": et_rmse,
            "R2": et_r2,
        }
    )


# ============================================================
# 結果まとめ
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE_kmh").reset_index(drop=True)


print()
print("=" * 60)
print("特徴量セット比較結果")
print("=" * 60)

print()

print(results_df.to_string(index=False))


comparison_path = OUTPUT_DIR / "train839_feature_set_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 最良モデルを決定
# ============================================================

best_row = results_df.iloc[0]

best_feature_set = best_row["feature_set"]

best_model_name = best_row["model"]

best_mae = best_row["MAE_kmh"]
best_rmse = best_row["RMSE_kmh"]
best_r2 = best_row["R2"]


print()
print("=" * 60)
print("最良モデル")
print("=" * 60)

print()
print("特徴量セット:", best_feature_set)
print("モデル:", best_model_name)
print(f"MAE  : {best_mae:.3f} km/h")
print(f"RMSE : {best_rmse:.3f} km/h")
print(f"R²   : {best_r2:.3f}")


best_features = feature_sets[best_feature_set]

print()
print("使用特徴量:")

for feature in best_features:
    print(" -", feature)


# ============================================================
# 最良モデルを全データで再学習
# ============================================================

print()
print("=" * 60)
print("最終モデル学習")
print("=" * 60)


X_best = X_all[best_features].copy()


if best_model_name == "RandomForest":

    final_model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )

else:

    final_model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )


print()
print(f"{best_model_name}を全データで学習中...")

final_model.fit(X_best, y)

print("学習完了")


# ============================================================
# モデル保存
# ============================================================

final_model_path = OUTPUT_DIR / "best_train839_speed_model.pkl"

joblib.dump(final_model, final_model_path)


# ============================================================
# モデル情報保存
# ============================================================

model_info = {
    "model": best_model_name,
    "feature_set": best_feature_set,
    "features": best_features,
    "videos": len(y),
    "mae_kmh_loocv": float(best_mae),
    "rmse_kmh_loocv": float(best_rmse),
    "r2_loocv": float(best_r2),
}


model_info_path = OUTPUT_DIR / "best_model_info.txt"


with open(model_info_path, "w", encoding="utf-8") as f:

    f.write("839本 球速推定 最良モデル\n")

    f.write("=" * 50 + "\n\n")

    f.write(f"モデル: {best_model_name}\n")

    f.write(f"特徴量セット: {best_feature_set}\n")

    f.write(f"動画数: {len(y)}\n")

    f.write(f"LOOCV MAE: {best_mae:.6f} km/h\n")

    f.write(f"LOOCV RMSE: {best_rmse:.6f} km/h\n")

    f.write(f"LOOCV R2: {best_r2:.6f}\n\n")

    f.write("使用特徴量:\n")

    for feature in best_features:

        f.write(f"- {feature}\n")


# ============================================================
# 完了
# ============================================================

print()
print("=" * 60)
print("解析完了")
print("=" * 60)

print()
print("比較結果:")
print(comparison_path)

print()
print("特徴量重要度:")
print(importance_path)

print()
print("最終モデル:")
print(final_model_path)

print()
print("モデル情報:")
print(model_info_path)

print()
print("=" * 60)
print("完了")
print("=" * 60)
