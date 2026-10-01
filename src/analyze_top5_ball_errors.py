import pandas as pd

prediction_file = "output/extratrees_ball_speed_predictions_40.csv"
feature_file = "output/ball_tracking_dataset_40.csv"

pred = pd.read_csv(prediction_file)
features = pd.read_csv(feature_file)

top5 = pred.sort_values("absolute_error_kmh", ascending=False).head(5)

result = top5.merge(features, on="video", how="left", suffixes=("", "_feature"))

columns = [
    "video",
    "actual_speed_kmh",
    "predicted_speed_kmh",
    "error_kmh",
    "absolute_error_kmh",
    "num_frames",
    "average_pixel_speed",
    "max_pixel_speed",
    "path_straightness",
    "total_distance_px",
    "direction_change",
    "mean_confidence",
    "min_confidence",
]

print(result[columns].to_string(index=False))

result[columns].to_csv(
    "output/top5_ball_error_analysis.csv", index=False, encoding="utf-8-sig"
)

print("\n分析結果を保存しました：")
print("output/top5_ball_error_analysis.csv")
