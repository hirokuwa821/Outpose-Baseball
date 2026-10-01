from pathlib import Path
import pandas as pd

INPUT_FILE = Path("output/unique_videos.csv")
OUTPUT_FILE = Path("output/ball_label_candidates_20.csv")

df = pd.read_csv(INPUT_FILE)

# 前回選出した20本を除外
previous = pd.read_csv("output/ball_label_candidates_20.csv")
df = df[~df["video"].isin(previous["video"])].copy()

# 球速がある動画だけ
df = df.dropna(subset=["speed_kmh"]).copy()

# 球速帯
bins = [130, 140, 150, 160, 170]
labels = ["130-139", "140-149", "150-159", "160-169"]

df["speed_band"] = pd.cut(df["speed_kmh"], bins=bins, labels=labels, right=False)

# 各球速帯から5本ずつ選ぶ
selected = []

for band in labels:
    band_df = df[df["speed_band"] == band]

    n = min(5, len(band_df))

    # ランダムに選ぶ
    sample = band_df.sample(n=n, random_state=42)

    selected.append(sample)

result = pd.concat(selected)

# 球速順に並べる
result = result.sort_values("speed_kmh")

# 保存
result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("=" * 60)
print("追加ラベル用動画を選出しました")
print("=" * 60)

print()
print(f"選出本数 : {len(result)}本")

print()
print("球速帯ごとの本数")
print("-" * 60)

print(result["speed_band"].value_counts().sort_index().to_string())

print()
print("選出された動画")
print("-" * 60)

for _, row in result.iterrows():
    print(f"{row['speed_kmh']:.2f} km/h  " f"{row['video']}")

print()
print("保存先:")
print(OUTPUT_FILE)
