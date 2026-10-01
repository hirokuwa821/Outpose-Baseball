import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("output/ball_feature_dataset.csv")
OUTPUT_FILE = Path("output/ball_feature_dataset_labeled.csv")


def extract_speed_mph(video_name):
    """
    ファイル名から球速(mph)を取得する
    例:
    mlb_001_81.5mph_xxx.mp4
    → 81.5
    """
    match = re.search(r"_(\d+(?:\.\d+)?)mph_", video_name)

    if match:
        return float(match.group(1))

    return None


# CSV読み込み
df = pd.read_csv(INPUT_FILE)

# mphを取得
df["speed_mph"] = df["video"].apply(extract_speed_mph)

# mph → km/h
df["speed_kmh"] = df["speed_mph"] * 1.609344

# 保存
df.to_csv(OUTPUT_FILE, index=False)

print("球速ラベルの追加が完了しました")
print()

print(df[["video", "speed_mph", "speed_kmh"]].to_string(index=False))

print()
print("保存先:", OUTPUT_FILE)
