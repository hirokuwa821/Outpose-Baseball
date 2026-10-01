from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 設定
# ============================================================

INPUT_CSV = Path("output/ball_tracking_dataset_train11_839.csv")

OUTPUT_DIR = Path("output/train839_speed_band_model")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 今回使用する特徴量
FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]


# 球速帯
def speed_band(speed):

    if speed < 140:
        return "130-139"
    elif speed < 150:
        return "140-149"
    elif speed < 160:
        return "150-159"
    else:
        return "160-169"


BANDS = [
    "130-139",
    "140-149",
    "150-159",
    "160-169",
]


# ============================================================
# 開始
# ============================================================

print("=" * 60)
print("839本 2段階球速推定モデル")
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
# 必要列確認
# ============================================================

required_columns = FEATURES + ["speed_kmh"]

missing = [c for c in required_columns if c not in df.columns]

if missing:

    print()
    print("エラー")
    print("必要な列がありません:")

    for c in missing:
        print(" -", c)

    raise ValueError("必要な列がCSVにありません")


# ============================================================
# 欠損値処理
# ============================================================

print()
print("=" * 60)
print("欠損値処理")
print("=" * 60)

before = len(df)

df = df.dropna(subset=required_columns).copy()

after = len(df)

print("処理前:", before)
print("処理後:", after)
print("除外:", before - after)


# ============================================================
# 球速帯作成
# ============================================================

df["speed_band"] = df["speed_kmh"].apply(speed_band)


# ============================================================
# 球速帯統計
# ============================================================

print()
print("=" * 60)
print("球速帯")
print("=" * 60)

print(df["speed_band"].value_counts().reindex(BANDS, fill_value=0).to_string())


# ============================================================
# 特徴量
# ============================================================

X = df[FEATURES].copy()

y_band = df["speed_band"].copy()

y_speed = df["speed_kmh"].copy()


# ============================================================
# ① 球速帯分類モデル
# ============================================================

print()
print("=" * 60)
print("① 球速帯分類モデル")
print("=" * 60)

classifier = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
)


# ============================================================
# LOOCV分類
# ============================================================

loo = LeaveOneOut()

band_predictions = []
band_actual = []

print()
print("LOOCV実行中...")


for train_index, test_index in loo.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y_band.iloc[train_index]
    y_test = y_band.iloc[test_index]

    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_test)[0]

    band_predictions.append(prediction)

    band_actual.append(y_test.iloc[0])


band_accuracy = accuracy_score(band_actual, band_predictions)


print()
print("球速帯分類精度:")
print(f"{band_accuracy * 100:.2f}%")


# ============================================================
# 分類レポート
# ============================================================

print()
print("=" * 60)
print("分類レポート")
print("=" * 60)

print(
    classification_report(
        band_actual,
        band_predictions,
        labels=BANDS,
        zero_division=0,
    )
)


# ============================================================
# 混同行列
# ============================================================

cm = confusion_matrix(
    band_actual,
    band_predictions,
    labels=BANDS,
)


cm_df = pd.DataFrame(
    cm,
    index=[f"actual_{b}" for b in BANDS],
    columns=[f"predicted_{b}" for b in BANDS],
)


cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")


# ============================================================
# ② 各球速帯専用の回帰モデル
# ============================================================

print()
print("=" * 60)
print("② 球速帯別回帰モデル")
print("=" * 60)


band_models = {}


for band in BANDS:

    band_df = df[df["speed_band"] == band].copy()

    print()
    print(f"{band} km/h帯:")

    print("データ数:", len(band_df))

    if len(band_df) < 5:

        print("データ不足のためスキップ")

        continue

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    )

    model.fit(band_df[FEATURES], band_df["speed_kmh"])

    band_models[band] = model


# ============================================================
# 2段階LOOCV
# ============================================================

print()
print("=" * 60)
print("2段階LOOCV")
print("=" * 60)

two_stage_predictions = []

actual_speeds = []

predicted_bands = []


print()
print("LOOCV実行中...")


for train_index, test_index in loo.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_band_train = y_band.iloc[train_index]
    y_speed_train = y_speed.iloc[train_index]

    actual_speed = y_speed.iloc[test_index[0]]

    # --------------------------------------------------------
    # Step 1: 球速帯分類
    # --------------------------------------------------------

    classifier_cv = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    )

    classifier_cv.fit(X_train, y_band_train)

    predicted_band = classifier_cv.predict(X_test)[0]

    # --------------------------------------------------------
    # Step 2: その球速帯の回帰モデル
    # --------------------------------------------------------

    band_mask = y_band_train == predicted_band

    X_band = X_train[band_mask]

    y_band_speed = y_speed_train[band_mask]

    # データが少なすぎる場合は全体モデルを使用
    if len(X_band) < 5:

        regression_model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        )

        regression_model.fit(X_train, y_speed_train)

    else:

        regression_model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        )

        regression_model.fit(X_band, y_band_speed)

    predicted_speed = regression_model.predict(X_test)[0]

    two_stage_predictions.append(predicted_speed)

    actual_speeds.append(actual_speed)

    predicted_bands.append(predicted_band)


# ============================================================
# 評価
# ============================================================

actual_speeds = np.array(actual_speeds)

two_stage_predictions = np.array(two_stage_predictions)


mae = mean_absolute_error(actual_speeds, two_stage_predictions)

rmse = np.sqrt(mean_squared_error(actual_speeds, two_stage_predictions))

r2 = r2_score(actual_speeds, two_stage_predictions)


print()
print("=" * 60)
print("2段階モデル結果")
print("=" * 60)

print()
print(f"MAE  : {mae:.3f} km/h")

print(f"RMSE : {rmse:.3f} km/h")

print(f"R²   : {r2:.3f}")


# ============================================================
# 予測結果保存
# ============================================================

result = df.copy()

result["predicted_band"] = predicted_bands

result["predicted_speed_kmh"] = two_stage_predictions

result["error_kmh"] = two_stage_predictions - actual_speeds

result["absolute_error_kmh"] = np.abs(result["error_kmh"])


result.to_csv(
    OUTPUT_DIR / "two_stage_predictions.csv", index=False, encoding="utf-8-sig"
)


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


# ============================================================
# 誤差範囲
# ============================================================

print()
print("=" * 60)
print("誤差範囲")
print("=" * 60)


abs_error = result["absolute_error_kmh"]


ranges = {
    "0-2": (abs_error < 2),
    "2-5": ((abs_error >= 2) & (abs_error < 5)),
    "5-10": ((abs_error >= 5) & (abs_error < 10)),
    "10-15": ((abs_error >= 10) & (abs_error < 15)),
    "15+": (abs_error >= 15),
}


error_rows = []


for name, mask in ranges.items():

    count = int(mask.sum())

    percentage = count / len(result) * 100

    error_rows.append(
        {
            "error_range": name,
            "count": count,
            "percentage": percentage,
        }
    )

    print(f"{name:>5} km/h : " f"{count:4d}本 " f"({percentage:.1f}%)")


error_df = pd.DataFrame(error_rows)


error_df.to_csv(OUTPUT_DIR / "error_ranges.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 球速帯別性能
# ============================================================

print()
print("=" * 60)
print("球速帯別性能")
print("=" * 60)


group_rows = []


for band in BANDS:

    mask = result["speed_band"] == band

    group = result[mask]

    if len(group) == 0:
        continue

    group_mae = mean_absolute_error(group["speed_kmh"], group["predicted_speed_kmh"])

    group_rmse = np.sqrt(
        mean_squared_error(group["speed_kmh"], group["predicted_speed_kmh"])
    )

    group_rows.append(
        {
            "speed_band": band,
            "videos": len(group),
            "MAE_kmh": group_mae,
            "RMSE_kmh": group_rmse,
            "actual_mean": group["speed_kmh"].mean(),
            "predicted_mean": group["predicted_speed_kmh"].mean(),
        }
    )

    print()
    print(band)

    print("動画数:", len(group))

    print(f"MAE : {group_mae:.3f} km/h")

    print(f"RMSE: {group_rmse:.3f} km/h")


group_df = pd.DataFrame(group_rows)


group_df.to_csv(
    OUTPUT_DIR / "speed_group_performance.csv", index=False, encoding="utf-8-sig"
)


# ============================================================
# 予測値範囲
# ============================================================

print()
print("=" * 60)
print("予測値分布")
print("=" * 60)

print()

print("実測最低:", f"{actual_speeds.min():.3f}")

print("実測最高:", f"{actual_speeds.max():.3f}")

print("予測最低:", f"{two_stage_predictions.min():.3f}")

print("予測最高:", f"{two_stage_predictions.max():.3f}")

print("実測平均:", f"{actual_speeds.mean():.3f}")

print("予測平均:", f"{two_stage_predictions.mean():.3f}")


# ============================================================
# 最終モデル作成
# ============================================================

print()
print("=" * 60)
print("最終モデル作成")
print("=" * 60)


# 分類モデル
final_classifier = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
)


final_classifier.fit(X, y_band)


# 球速帯ごとの回帰モデル
final_regressors = {}


for band in BANDS:

    band_mask = y_band == band

    X_band = X[band_mask]

    y_band_speed = y_speed[band_mask]

    if len(X_band) < 5:
        continue

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    )

    model.fit(X_band, y_band_speed)

    final_regressors[band] = model


# ============================================================
# モデル保存
# ============================================================

import joblib

model_package = {
    "classifier": final_classifier,
    "regressors": final_regressors,
    "features": FEATURES,
    "bands": BANDS,
}


MODEL_PATH = OUTPUT_DIR / "train839_two_stage_model.pkl"


joblib.dump(model_package, MODEL_PATH)


print()
print("モデル保存:")
print(MODEL_PATH)


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

print(" - two_stage_predictions.csv")

print(" - error_ranges.csv")

print(" - speed_group_performance.csv")

print(" - confusion_matrix.csv")

print(" - train839_two_stage_model.pkl")

print()
print("=" * 60)
