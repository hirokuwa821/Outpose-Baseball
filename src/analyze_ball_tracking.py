import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "output/ball_tracking_dataset_20.csv"

df = pd.read_csv(INPUT_CSV)

# 追跡成功した動画だけ
df = df[df["track_success"] == 1].copy()

print("=" * 60)
print("ボール追跡データ分析")
print("=" * 60)

print("追跡成功動画数:", len(df))
print()

# 球速との相関を見る特徴量
feature_cols = [
    "num_frames",
    "duration_sec",
    "total_distance_px",
    "start_end_distance_px",
    "average_pixel_speed",
    "max_pixel_speed",
    "total_dx",
    "total_dy",
    "direction_change",
    "mean_confidence",
    "min_confidence",
]

# 相関係数
corr = df[["speed_kmh"] + feature_cols].corr()["speed_kmh"]
corr = corr.drop("speed_kmh")
corr = corr.sort_values(key=abs, ascending=False)

print("【球速との相関係数】")
print("-" * 60)

for feature, value in corr.items():
    print(f"{feature:25s}: {value:+.3f}")

print()

# CSVにも保存
corr_df = corr.reset_index()
corr_df.columns = ["feature", "correlation"]

corr_df.to_csv(
    "output/ball_feature_correlations_20.csv", index=False, encoding="utf-8-sig"
)

print("相関結果を保存:")
print("output/ball_feature_correlations_20.csv")

# ==========================================
# 平均画面速度 vs 実測球速
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(df["average_pixel_speed"], df["speed_kmh"])

plt.xlabel("Average pixel speed")
plt.ylabel("Actual pitch speed (km/h)")
plt.title("Average Pixel Speed vs Actual Pitch Speed")

plt.grid(True)
plt.tight_layout()

plt.savefig("output/average_pixel_speed_vs_pitch_speed.png", dpi=200)

plt.show()

print()
print("グラフ保存:")
print("output/average_pixel_speed_vs_pitch_speed.png")
