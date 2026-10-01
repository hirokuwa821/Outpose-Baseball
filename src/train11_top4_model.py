import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 設定
# ==========================================

INPUT_CSV = "output/ball_tracking_dataset_train11_40.csv"

OUTPUT_MODEL = "output/train11_top4_model.pkl"

TARGET = "speed_kmh"


# ==========================================
# TOP_4特徴量
# ==========================================

FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]


# ==========================================
# データ読み込み
# ==========================================

df = pd.read_csv(INPUT_CSV)


# ==========================================
# 追跡成功データのみ使用
# ==========================================

df = df[df["track_success"] == 1].copy()


# ==========================================
# 欠損値除外
# ==========================================

df = df.dropna(subset=FEATURES + [TARGET])


# ==========================================
# X / y
# ==========================================

X = df[FEATURES]

y = df[TARGET]


# ==========================================
# 基本情報
# ==========================================

print("=" * 60)
print("train-11 TOP_4 球速推定モデル")
print("=" * 60)

print()

print("データ数:", len(df))

print("特徴量数:", len(FEATURES))

print()

print("使用特徴量:")

for feature in FEATURES:

    print(" -", feature)

print()

print("球速範囲:")

print(f"最低: {y.min():.3f} km/h")

print(f"最高: {y.max():.3f} km/h")

print(f"平均: {y.mean():.3f} km/h")


# ==========================================
# Random Forest
# ==========================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_features=0.8,
    min_samples_leaf=2,
    n_jobs=-1,
)


# ==========================================
# 全データで学習
# ==========================================

print()

print("=" * 60)
print("モデル学習")
print("=" * 60)

print()

print("学習開始...")


model.fit(X, y)


print("学習完了")


# ==========================================
# 学習データに対する予測
# ==========================================

predictions = model.predict(X)


# ==========================================
# 学習データ上の誤差
# ==========================================

mae = np.mean(np.abs(predictions - y))

rmse = np.sqrt(np.mean((predictions - y) ** 2))


print()

print("=" * 60)
print("学習データ上の結果")
print("=" * 60)

print()

print(f"MAE  : {mae:.3f} km/h")

print(f"RMSE : {rmse:.3f} km/h")


print()

print("※この値は学習に使用したデータに対する")

print("  誤差なので、未知データの性能を表すものではありません。")


# ==========================================
# モデル情報
# ==========================================

model_data = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "training_videos": len(df),
}


# ==========================================
# 保存
# ==========================================

Path(OUTPUT_MODEL).parent.mkdir(parents=True, exist_ok=True)


with open(OUTPUT_MODEL, "wb") as f:

    pickle.dump(model_data, f)


# ==========================================
# 保存確認
# ==========================================

print()

print("=" * 60)
print("モデル保存完了")
print("=" * 60)

print()

print("保存先:")

print(OUTPUT_MODEL)

print()

print("使用特徴量:")

for feature in FEATURES:

    print(" -", feature)

print()

print("このモデルは、次の4特徴量を入力すると")

print("球速[km/h]を推定できます。")

print()

print("モデル作成完了")
