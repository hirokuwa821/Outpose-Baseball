from pathlib import Path
import pandas as pd

INPUT_FILE = Path("output/unique_videos.csv")
OUTPUT_FILE = Path("output/unique_videos_with_speed.csv")

print("=" * 60)
print("839本 学習用動画リスト作成")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print()
print("読み込み動画数:", len(df))

required_columns = ["video", "speed_mph", "speed_kmh"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    print("必要な列がありません:")
    for c in missing_columns:
        print(" -", c)
    raise ValueError("unique_videos.csv の列を確認してください")

before_count = len(df)

df = df.dropna(subset=["speed_mph", "speed_kmh"]).copy()

print()
print("=" * 60)
print("球速ラベル確認")
print("=" * 60)
print("処理前:", before_count)
print("球速あり:", len(df))
print("球速なし:", before_count - len(df))

duplicate_videos = df["video"].duplicated().sum()

print()
print("=" * 60)
print("重複確認")
print("=" * 60)
print("動画名の重複:", duplicate_videos)

if duplicate_videos > 0:
    df = df.drop_duplicates(subset=["video"], keep="first").copy()

print()
print("=" * 60)
print("球速統計")
print("=" * 60)
print(f"最低: {df['speed_kmh'].min():.3f} km/h")
print(f"最高: {df['speed_kmh'].max():.3f} km/h")
print(f"平均: {df['speed_kmh'].mean():.3f} km/h")
print(f"中央値: {df['speed_kmh'].median():.3f} km/h")

bins = [0, 130, 140, 150, 160, 170, 180, float("inf")]
labels = [
    "<130",
    "130-139.9",
    "140-149.9",
    "150-159.9",
    "160-169.9",
    "170-179.9",
    "180+",
]

speed_bins = pd.cut(df["speed_kmh"], bins=bins, labels=labels, right=False)

print()
print("=" * 60)
print("球速区間")
print("=" * 60)
print(speed_bins.value_counts(sort=False).to_string())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 60)
print("学習用動画リスト作成完了")
print("=" * 60)
print()
print("最終動画数:", len(df))
print()
print("保存先:")
print(OUTPUT_FILE)
