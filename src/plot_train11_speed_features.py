import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_DIR = Path("output/train11_speed_feature_plots")

TARGET = "speed_kmh"

FEATURES = [
    "std_dy",
    "min_confidence",
    "std_pixel_speed",
    "trajectory_angle_deg",
    "total_dy",
    "std_step_distance",
    "std_dx",
    "max_pixel_speed",
    "average_pixel_speed",
    "total_distance_px",
]


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

df = df.dropna(subset=FEATURES + [TARGET]).copy()

print("=" * 60)
print("train-11 球速・特徴量散布図")
print("=" * 60)

print()
print("データ数:", len(df))
print()


# ==========================================
# 保存先
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 散布図作成
# ==========================================

for feature in FEATURES:

    plt.figure(figsize=(7, 5))

    plt.scatter(df[feature], df[TARGET], alpha=0.8)

    plt.xlabel(feature)
    plt.ylabel("Actual speed (km/h)")

    plt.title(f"Actual Speed vs {feature}")

    plt.grid(True, alpha=0.3)

    output_path = OUTPUT_DIR / f"speed_vs_{feature}.png"

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)

    plt.close()

    print("保存:", output_path)


print()
print("=" * 60)
print("完了")
print("=" * 60)
