import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

INPUT = "output/ball_tracking_dataset_40.csv"

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

# 相関結果から選んだ特徴量
TOP10 = [
    "std_dy",
    "total_dy",
    "trajectory_angle_deg",
    "mean_dy",
    "direction_change",
    "path_straightness",
    "num_frames",
    "duration_sec",
    "std_dx",
    "total_distance_px",
]

FEATURE_SETS = {
    "std_dyのみ": TOP10[:1],
    "上位3個": TOP10[:3],
    "上位5個": TOP10[:5],
    "上位10個": TOP10[:10],
    "22個全部": FEATURES,
}

df = pd.read_csv(INPUT)

X_all = df
y = df["speed_kmh"]

loo = LeaveOneOut()

results = []

for name, features in FEATURE_SETS.items():

    X = df[features]

    predictions = []
    actuals = []

    for train_idx, test_idx in loo.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        predictions.append(pred)
        actuals.append(y_test.iloc[0])

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    results.append(
        {
            "feature_set": name,
            "num_features": len(features),
            "MAE_kmh": mae,
            "RMSE_kmh": rmse,
            "R2": r2,
        }
    )

result_df = pd.DataFrame(results)

print()
print("特徴量セット別 ExtraTrees 比較")
print(result_df.to_string(index=False))

result_df.to_csv(
    "output/ball_feature_set_comparison.csv", index=False, encoding="utf-8-sig"
)

print()
print("結果保存: output/ball_feature_set_comparison.csv")
