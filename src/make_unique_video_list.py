from pathlib import Path
import pandas as pd

INPUT_FILE = Path("output/video_duplicate_check.csv")
OUTPUT_FILE = Path("output/unique_videos.csv")
DUPLICATE_FILE = Path("output/duplicate_groups.csv")


# CSV読み込み
df = pd.read_csv(INPUT_FILE)

print("元の動画数:", len(df))

# --------------------------------------------------
# 1. 重複グループごとに代表動画を1本だけ残す
# --------------------------------------------------

unique_df = df.sort_values("video").drop_duplicates(subset="hash", keep="first").copy()

# --------------------------------------------------
# 2. 重複グループ番号を整理
# --------------------------------------------------

unique_df["duplicate_group"] = unique_df["hash"].map(
    df.drop_duplicates("hash").set_index("hash")["duplicate_group"]
)

# 重複しているか
hash_counts = df["hash"].value_counts()

unique_df["was_duplicate"] = unique_df["hash"].map(hash_counts) > 1

# --------------------------------------------------
# 3. 球速をファイル名から取得
# --------------------------------------------------

import re


def extract_speed_mph(video_name):
    match = re.search(r"_(\d+(?:\.\d+)?)mph(?:_|\.mp4)", video_name)

    if match:
        return float(match.group(1))

    return None


unique_df["speed_mph"] = unique_df["video"].apply(extract_speed_mph)

unique_df["speed_kmh"] = unique_df["speed_mph"] * 1.609344

# --------------------------------------------------
# 4. 左右投げ情報を取得
# --------------------------------------------------


def extract_hand(video_name):
    name = video_name.lower()

    if "_left" in name:
        return "left"

    if "_right" in name:
        return "right"

    return "unknown"


unique_df["hand"] = unique_df["video"].apply(extract_hand)

# --------------------------------------------------
# 5. 保存
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

unique_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 6. 重複グループ一覧も保存
# --------------------------------------------------

duplicate_df = df[df["is_duplicate"]].copy()

duplicate_df.to_csv(DUPLICATE_FILE, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# 結果表示
# --------------------------------------------------

print()
print("=" * 60)
print("ユニーク動画一覧を作成しました")
print("=" * 60)

print("元の動画数        :", len(df))
print("ユニーク動画数    :", len(unique_df))
print("重複を除いて削減  :", len(df) - len(unique_df))

print()
print("球速あり          :", unique_df["speed_mph"].notna().sum())

print("球速なし          :", unique_df["speed_mph"].isna().sum())

print()
print("左右投げ:")

print(unique_df["hand"].value_counts().to_string())

print()
print("保存先:")
print(OUTPUT_FILE)

print()
print("重複グループ保存先:")
print(DUPLICATE_FILE)
