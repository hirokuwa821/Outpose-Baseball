import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# 設定
# ============================================================

INPUT_CSV = "output/ball_tracking_dataset_train11_839.csv"

OUTPUT_DIR = Path("output")


FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]

TARGET = "speed_kmh"


# ============================================================
# データ読み込み
# ============================================================

print("=" * 60)
print("839本 TOP_4 球速推定モデル")
print("=" * 60)

print()
print("入力CSV:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print()
print("読み込みデータ数:", len(df))


# ============================================================
# 必要列確認
# ============================================================

required_columns = FEATURES + [TARGET]

missing = [column for column in required_columns if column not in df.columns]

if missing:

    print()
    print("エラー:")
    print("必要な列がありません")

    for column in missing:
        print(" -", column)

    raise ValueError("必要な特徴量または正解値がありません")


# ============================================================
# 欠損値除去
# ============================================================

before = len(df)

df = df.dropna(subset=required_columns).copy()

after = len(df)

print()
print("欠損値処理")
print("処理前:", before)
print("処理後:", after)
print("除外:", before - after)


# ============================================================
# X / y
# ============================================================

X = df[FEATURES]

y = df[TARGET]


print()
print("=" * 60)
print("学習データ")
print("=" * 60)

print()
print("動画数:", len(df))
print("特徴量数:", len(FEATURES))

print()
print("使用特徴量:")

for feature in FEATURES:
    print(" -", feature)


print()
print("球速範囲")
print("最低:", f"{y.min():.3f} km/h")

print("最高:", f"{y.max():.3f} km/h")

print("平均:", f"{y.mean():.3f} km/h")

print("中央値:", f"{y.median():.3f} km/h")


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
# Leave-One-Out
# ============================================================

cv = LeaveOneOut()


results = []


# ============================================================
# モデル評価
# ============================================================

for model_name, model in models.items():

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print()
    print("LOOCV実行中...")

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
    print("MAE :")
    print(f"{mae:.3f} km/h")

    print("RMSE :")
    print(f"{rmse:.3f} km/h")

    print("R² :")
    print(f"{r2:.3f}")

    # --------------------------------------------------------
    # 結果保存
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

    prediction_output = OUTPUT_DIR / f"{model_name.lower()}_predictions_839_top4.csv"

    prediction_df.to_csv(
        prediction_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("予測結果保存:")
    print(prediction_output)

    # --------------------------------------------------------
    # 結果追加
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
# 比較結果
# ============================================================

results_df = pd.DataFrame(results)


comparison_output = OUTPUT_DIR / "ball_speed_model_comparison_839_top4.csv"


results_df.to_csv(
    comparison_output,
    index=False,
    encoding="utf-8-sig",
)


print()
print("=" * 60)
print("モデル比較結果")
print("=" * 60)

print()

print(results_df.to_string(index=False))


print()
print("保存先:")
print(comparison_output)


# ============================================================
# 最終モデル学習
# ============================================================

print()
print("=" * 60)
print("最終モデル学習")
print("=" * 60)


final_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_features=0.8,
    min_samples_leaf=2,
    n_jobs=-1,
)


print()
print("RandomForestを全データで学習中...")


final_model.fit(X, y)


print("学習完了")


# ============================================================
# モデル保存
# ============================================================

model_output = OUTPUT_DIR / "train839_top4_randomforest.pkl"


joblib.dump(final_model, model_output)


print()
print("モデル保存:")
print(model_output)


print()
print("=" * 60)
print("完了")
print("=" * 60)
