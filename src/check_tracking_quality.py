import pandas as pd
from pathlib import Path

TRACKING_DIR = Path("output/ball_tracking_v2")

results = []

for csv_path in TRACKING_DIR.glob("*_tracking.csv"):
    df = pd.read_csv(csv_path)

    if len(df) < 2:
        continue

    x = df["center_x"]
    y = df["center_y"]
    conf = df["confidence"]

    dx = x.diff().dropna()
    dy = y.diff().dropna()

    total_distance = ((dx**2 + dy**2) ** 0.5).sum()
    start_end_distance = (
        (x.iloc[-1] - x.iloc[0]) ** 2 + (y.iloc[-1] - y.iloc[0]) ** 2
    ) ** 0.5

    straightness = start_end_distance / total_distance if total_distance > 0 else 0

    results.append(
        {
            "video": csv_path.name,
            "frames": len(df),
            "start_frame": int(df["frame"].iloc[0]),
            "end_frame": int(df["frame"].iloc[-1]),
            "mean_conf": conf.mean(),
            "min_conf": conf.min(),
            "straightness": straightness,
            "x_range": x.max() - x.min(),
            "y_range": y.max() - y.min(),
        }
    )

result_df = pd.DataFrame(results)

# 要注意条件
result_df["warning"] = (
    (result_df["frames"] <= 7)
    | (result_df["mean_conf"] < 0.45)
    | (result_df["straightness"] < 0.85)
)

warnings = result_df[result_df["warning"]].copy()

warnings = warnings.sort_values(["straightness", "mean_conf"])

print()
print("=== 追跡品質 要注意動画 ===")
print(
    warnings[
        [
            "video",
            "frames",
            "start_frame",
            "end_frame",
            "mean_conf",
            "min_conf",
            "straightness",
        ]
    ].to_string(index=False)
)

print()
print(f"全動画数: {len(result_df)}")
print(f"要注意動画数: {len(warnings)}")

result_df.to_csv("output/ball_tracking_quality.csv", index=False, encoding="utf-8-sig")

print("結果保存: output/ball_tracking_quality.csv")
