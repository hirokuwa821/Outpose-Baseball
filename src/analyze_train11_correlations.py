import pandas as pd
from pathlib import Path

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_CSV = "output/train11_feature_correlations.csv"

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

TARGET = "speed_kmh"


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

print("=" * 60)
print("train-11 特徴量・球速 相関解析")
print("=" * 60)
print()

print("読み込みデータ数:", len(df))

df = df.dropna(subset=FEATURES + [TARGET]).copy()

print("解析対象データ数:", len(df))
print()


# ==========================================
# 球速との相関
# ==========================================

correlations = []

for feature in FEATURES:

    correlation = df[feature].corr(df[TARGET])

    correlations.append(
        {
            "feature": feature,
            "correlation_with_speed": correlation,
            "absolute_correlation": abs(correlation),
        }
    )


correlation_df = pd.DataFrame(correlations)

correlation_df = correlation_df.sort_values(
    "absolute_correlation", ascending=False
).reset_index(drop=True)


# ==========================================
# 表示
# ==========================================

print("=" * 60)
print("実測球速との相関")
print("=" * 60)
print()

print(
    correlation_df[
        [
            "feature",
            "correlation_with_speed",
            "absolute_correlation",
        ]
    ].to_string(index=False)
)


# ==========================================
# 相関の解釈
# ==========================================

print()
print("=" * 60)
print("相関の目安")
print("=" * 60)
print()

print("|r| < 0.2   : ほぼ弱い相関")
print("0.2～0.4    : 弱い相関")
print("0.4～0.6    : 中程度の相関")
print("0.6～0.8    : 強い相関")
print("0.8以上     : 非常に強い相関")


# ==========================================
# 保存
# ==========================================

Path("output").mkdir(parents=True, exist_ok=True)

correlation_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


print()
print("=" * 60)
print("保存完了")
print("=" * 60)
print()

print("保存先:")
print(OUTPUT_CSV)
