import pandas as pd
from scipy.stats import pearsonr, spearmanr

INPUT = "output/ball_tracking_dataset_40.csv"
OUTPUT = "output/ball_feature_correlations.csv"

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

df = pd.read_csv(INPUT)

results = []

for feature in FEATURES:
    x = df[feature]
    y = df["speed_kmh"]

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    results.append(
        {
            "feature": feature,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        }
    )

result_df = pd.DataFrame(results)

result_df["abs_pearson"] = result_df["pearson_r"].abs()
result_df = result_df.sort_values("abs_pearson", ascending=False)

result_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("球速との相関が強い特徴量 TOP10")
print(result_df[["feature", "pearson_r", "spearman_r"]].head(10).to_string(index=False))

print()
print("全結果保存:", OUTPUT)
