from pathlib import Path

import joblib
import pandas as pd

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "output/train839_top4_randomforest.pkl"

# 推定対象の追跡CSV
INPUT_CSV = "output/new_tracking.csv"

# 839本モデルで使用するTOP_4特徴量
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
print("839本 TOP_4 球速推定")
print("=" * 60)

print()
print("モデル:")
print(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# 入力CSV確認
# ============================================================

print()
print("=" * 60)
print("追跡CSV読み込み")
print("=" * 60)

print()
print("入力:")
print(INPUT_CSV)

input_path = Path(INPUT_CSV)

if not input_path.exists():
    print()
    print("入力CSVがありません。")
    print()
    print("現在は新しい動画がないため、")
    print("この段階ではエラーではありません。")
    print()
    print("新しい動画を処理した後に、")
    print("その追跡CSVを INPUT_CSV に指定してください。")
    raise SystemExit


df = pd.read_csv(INPUT_CSV)

print()
print("データ数:", len(df))


# ============================================================
# 必要特徴量確認
# ============================================================

print()
print("=" * 60)
print("特徴量確認")
print("=" * 60)

missing_features = [feature for feature in FEATURES if feature not in df.columns]

if missing_features:

    print()
    print("必要な特徴量がありません:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError("追跡CSVに必要な特徴量がありません。")


print()
print("使用特徴量:")

for feature in FEATURES:
    print(" -", feature)


# ============================================================
# 特徴量取得
# ============================================================

X = df[FEATURES].copy()


# ============================================================
# 欠損値確認
# ============================================================

print()
print("=" * 60)
print("欠損値確認")
print("=" * 60)

missing = X.isnull().sum()

if missing.sum() > 0:

    print()
    print("欠損値があります:")

    print(missing[missing > 0])

    print()
    print("中央値で補完します。")

    X = X.fillna(X.median())

else:

    print()
    print("欠損値なし")


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

    print(f"{i + 1}球目: " f"{speed:.2f} km/h")


# ============================================================
# 結果保存
# ============================================================

result = df.copy()

result["predicted_speed_kmh"] = prediction

output_path = input_path.parent / f"{input_path.stem}_predicted.csv"

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
print("球速推定完了")
print("=" * 60)

print()
print("結果保存:")
print(output_path)
