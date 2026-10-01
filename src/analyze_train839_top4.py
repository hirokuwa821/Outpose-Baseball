import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 設定
# ============================================================

INPUT_CSV = Path("output/randomforest_predictions_839_top4.csv")

DATASET_CSV = Path("output/ball_tracking_dataset_train11_839.csv")

MODEL_PATH = Path("output/train839_top4_randomforest.pkl")

OUTPUT_DIR = Path("output/train839_top4_analysis")

FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]

TARGET = "speed_kmh"


# ============================================================
# 出力フォルダ
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 開始
# ============================================================

print("=" * 60)
print("839本 TOP_4 モデル詳細解析")
print("=" * 60)


# ============================================================
# 予測結果読み込み
# ============================================================

print()
print("予測結果読み込み:")
print(INPUT_CSV)

prediction_df = pd.read_csv(INPUT_CSV)

print()
print("データ数:", len(prediction_df))


# ============================================================
# 列名確認
# ============================================================

required_prediction_columns = [
    "actual_speed_kmh",
    "predicted_speed_kmh",
]

for col in required_prediction_columns:

    if col not in prediction_df.columns:

        raise ValueError(f"必要な列がありません: {col}")


# ============================================================
# 誤差計算
# ============================================================

prediction_df["error_kmh"] = (
    prediction_df["predicted_speed_kmh"] - prediction_df["actual_speed_kmh"]
)

prediction_df["absolute_error_kmh"] = prediction_df["error_kmh"].abs()


# ============================================================
# 基本評価
# ============================================================

y_true = prediction_df["actual_speed_kmh"]
y_pred = prediction_df["predicted_speed_kmh"]


mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

r2 = r2_score(y_true, y_pred)


print()
print("=" * 60)
print("LOOCV評価")
print("=" * 60)

print()
print(f"MAE  : {mae:.3f} km/h")
print(f"RMSE : {rmse:.3f} km/h")
print(f"R²   : {r2:.3f}")


# ============================================================
# 誤差統計
# ============================================================

print()
print("=" * 60)
print("誤差統計")
print("=" * 60)

print()
print(f"平均絶対誤差 : " f"{prediction_df['absolute_error_kmh'].mean():.3f} km/h")

print(f"最大絶対誤差 : " f"{prediction_df['absolute_error_kmh'].max():.3f} km/h")

print(f"中央値絶対誤差 : " f"{prediction_df['absolute_error_kmh'].median():.3f} km/h")


# ============================================================
# 誤差区間
# ============================================================

print()
print("=" * 60)
print("誤差区間")
print("=" * 60)

abs_error = prediction_df["absolute_error_kmh"]

ranges = [
    ("0～2 km/h", 0, 2),
    ("2～5 km/h", 2, 5),
    ("5～10 km/h", 5, 10),
    ("10～15 km/h", 10, 15),
    ("15 km/h以上", 15, np.inf),
]

error_range_results = []

for name, low, high in ranges:

    count = ((abs_error >= low) & (abs_error < high)).sum()

    percentage = count / len(prediction_df) * 100

    print(f"{name:15s}: " f"{count:4d}本 " f"({percentage:.1f}%)")

    error_range_results.append(
        {
            "error_range": name,
            "videos": count,
            "percentage": percentage,
        }
    )


error_range_df = pd.DataFrame(error_range_results)

error_range_df.to_csv(
    OUTPUT_DIR / "error_ranges.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 球速帯ごとの誤差
# ============================================================

print()
print("=" * 60)
print("球速帯ごとの誤差")
print("=" * 60)


def speed_group(speed):

    if speed < 140:
        return "130-139.9"

    elif speed < 150:
        return "140-149.9"

    elif speed < 160:
        return "150-159.9"

    else:
        return "160-169.9"


prediction_df["speed_group"] = prediction_df["actual_speed_kmh"].apply(speed_group)


group_results = []


for group, group_df in prediction_df.groupby("speed_group", sort=True):

    group_mae = mean_absolute_error(
        group_df["actual_speed_kmh"], group_df["predicted_speed_kmh"]
    )

    group_rmse = np.sqrt(
        mean_squared_error(
            group_df["actual_speed_kmh"], group_df["predicted_speed_kmh"]
        )
    )

    group_results.append(
        {
            "speed_group": group,
            "videos": len(group_df),
            "MAE_kmh": group_mae,
            "RMSE_kmh": group_rmse,
            "mean_actual_kmh": group_df["actual_speed_kmh"].mean(),
            "mean_predicted_kmh": group_df["predicted_speed_kmh"].mean(),
        }
    )

    print()
    print(group)

    print("動画数:", len(group_df))

    print(f"MAE : {group_mae:.3f} km/h")

    print(f"RMSE: {group_rmse:.3f} km/h")

    print(f"実測平均: " f"{group_df['actual_speed_kmh'].mean():.3f}")

    print(f"予測平均: " f"{group_df['predicted_speed_kmh'].mean():.3f}")


group_results_df = pd.DataFrame(group_results)

group_results_df.to_csv(
    OUTPUT_DIR / "speed_group_performance.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 最大誤差ランキング
# ============================================================

print()
print("=" * 60)
print("最大誤差ランキング TOP20")
print("=" * 60)


worst_df = prediction_df.sort_values("absolute_error_kmh", ascending=False).head(20)


print(
    worst_df[
        [
            "video",
            "actual_speed_kmh",
            "predicted_speed_kmh",
            "error_kmh",
            "absolute_error_kmh",
        ]
    ].to_string(index=False)
)


worst_df.to_csv(
    OUTPUT_DIR / "worst_20_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 予測値の範囲確認
# ============================================================

print()
print("=" * 60)
print("予測値の分布")
print("=" * 60)

print()
print(f"実測最低 : " f"{y_true.min():.3f} km/h")

print(f"実測最高 : " f"{y_true.max():.3f} km/h")

print(f"予測最低 : " f"{y_pred.min():.3f} km/h")

print(f"予測最高 : " f"{y_pred.max():.3f} km/h")

print(f"実測平均 : " f"{y_true.mean():.3f} km/h")

print(f"予測平均 : " f"{y_pred.mean():.3f} km/h")


# ============================================================
# 予測結果完全版保存
# ============================================================

prediction_df.to_csv(
    OUTPUT_DIR / "predictions_with_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 特徴量重要度
# ============================================================

print()
print("=" * 60)
print("特徴量重要度")
print("=" * 60)


dataset_df = pd.read_csv(DATASET_CSV)

dataset_df = dataset_df.dropna(subset=FEATURES + [TARGET])

X = dataset_df[FEATURES]
y = dataset_df[TARGET]


model = joblib.load(MODEL_PATH)


print()
print("モデル:")
print(MODEL_PATH)

print()
print("Permutation Importance計算中...")


perm = permutation_importance(
    model,
    X,
    y,
    n_repeats=20,
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


importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
    encoding="utf-8-sig",
)


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
