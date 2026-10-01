import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "output/train11_top4_model.pkl"

# 推定したい追跡CSV
INPUT_CSV = "output/new_tracking.csv"

# TOP_4特徴量
FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]


# ============================================================
# モデル読み込み
# ============================================================

print("=" * 60)
print("TOP_4 球速推定")
print("=" * 60)

print()
print("モデル読み込み:")
print(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# CSV読み込み
# ============================================================

print()
print("=" * 60)
print("追跡CSV読み込み")
print("=" * 60)

print()
print("入力:")
print(INPUT_CSV)

df = pd.read_csv(INPUT_CSV)

print()
print("データ数:", len(df))


# ============================================================
# 必要特徴量の確認
# ============================================================

missing_features = [feature for feature in FEATURES if feature not in df.columns]

if missing_features:
    print()
    print("エラー")
    print("必要な特徴量がありません:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError("必要な特徴量がCSVにありません")


# ============================================================
# 特徴量取得
# ============================================================

X = df[FEATURES].copy()


# ============================================================
# 欠損値確認
# ============================================================

if X.isnull().any().any():

    print()
    print("警告: 欠損値があります")

    print(X.isnull().sum())

    X = X.fillna(X.median())


# ============================================================
# 球速推定
# ============================================================

print()
print("=" * 60)
print("球速推定")
print("=" * 60)

prediction = model.predict(X)


# ============================================================
# 結果表示
# ============================================================

print()

for i, speed in enumerate(prediction):

    print(f"推定球速: {speed:.2f} km/h")


# ============================================================
# 結果保存
# ============================================================

result = df.copy()

result["predicted_speed_kmh"] = prediction


output_path = Path(INPUT_CSV).parent / (Path(INPUT_CSV).stem + "_predicted.csv")


result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 完了
# ============================================================

print()
print("=" * 60)
print("推定完了")
print("=" * 60)

print()
print("結果保存:")
print(output_path)
