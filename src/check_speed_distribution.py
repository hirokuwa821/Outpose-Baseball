from pathlib import Path
import pandas as pd

INPUT_FILE = Path("output/unique_videos.csv")

df = pd.read_csv(INPUT_FILE)

# 球速がある動画だけ
speed_df = df.dropna(subset=["speed_kmh"]).copy()

print("=" * 60)
print("球速データの確認")
print("=" * 60)

print(f"全ユニーク動画数 : {len(df)}")
print(f"球速あり          : {len(speed_df)}")
print(f"球速なし          : {len(df) - len(speed_df)}")

print()
print("球速の基本統計")
print("-" * 60)

print(f"最低速度 : {speed_df['speed_kmh'].min():.2f} km/h")
print(f"最高速度 : {speed_df['speed_kmh'].max():.2f} km/h")
print(f"平均速度 : {speed_df['speed_kmh'].mean():.2f} km/h")
print(f"中央値   : {speed_df['speed_kmh'].median():.2f} km/h")

print()
print("球速帯ごとの本数")
print("-" * 60)

# km/hで球速帯を作る
bins = [120, 130, 140, 150, 160, 170, 180]

labels = [
    "120-129 km/h",
    "130-139 km/h",
    "140-149 km/h",
    "150-159 km/h",
    "160-169 km/h",
    "170-179 km/h",
]

speed_df["speed_band"] = pd.cut(
    speed_df["speed_kmh"], bins=bins, labels=labels, right=False
)

band_counts = speed_df["speed_band"].value_counts().sort_index()

for band, count in band_counts.items():
    print(f"{band:15s} : {count:4d}本")

print()
print("球速帯の割合")
print("-" * 60)

for band, count in band_counts.items():
    percentage = count / len(speed_df) * 100
    print(f"{band:15s} : {percentage:6.2f}%")

print()
print("右投げ・左投げ・不明")
print("-" * 60)

print(df["hand"].value_counts().to_string())

print()
print("球速なし動画")
print("-" * 60)

no_speed = df[df["speed_kmh"].isna()]

if len(no_speed) > 0:
    print(no_speed["video"].to_string(index=False))
else:
    print("なし")

print()
print("=" * 60)
print("確認完了")
print("=" * 60)
