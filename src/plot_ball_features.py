import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_40.csv"
OUTPUT_DIR = Path("output/ball_feature_graphs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)

# 追跡成功したデータのみ使用
df = df[df["track_success"] == 1].copy()

print("=" * 60)
print("球速と追跡特徴量の関係をグラフ化")
print("=" * 60)
print("使用動画数:", len(df))


# ==========================================
# グラフ作成関数
# ==========================================


def make_scatter_graph(feature, xlabel, filename):
    plt.figure(figsize=(8, 6))

    plt.scatter(df[feature], df["speed_kmh"], alpha=0.7)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Actual speed (km/h)", fontsize=12)

    plt.title(f"{xlabel} vs Actual Speed", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("保存:", output_path)


# ==========================================
# 3種類のグラフ
# ==========================================

make_scatter_graph(
    "total_distance_px", "Total distance (pixels)", "total_distance_vs_speed.png"
)

make_scatter_graph(
    "average_pixel_speed",
    "Average pixel speed (pixels/sec)",
    "average_pixel_speed_vs_speed.png",
)

make_scatter_graph("num_frames", "Number of tracked frames", "num_frames_vs_speed.png")


print()
print("グラフ作成完了")
