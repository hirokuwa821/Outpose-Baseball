from pathlib import Path
import subprocess
import sys

import pandas as pd
import joblib

# ============================================================
# 設定
# ============================================================

# 推定したい動画
INPUT_VIDEO = "test_videos/sample.mp4"

# 学習済みモデル
MODEL_PATH = "output/train839_top4_randomforest.pkl"

# 一時的に作成する追跡CSV
TRACKING_CSV = "output/predict_tracking.csv"


# TOP_4特徴量
FEATURES = [
    "std_dy",
    "path_straightness",
    "std_pixel_speed",
    "std_dx",
]


# ============================================================
# 開始
# ============================================================

print("=" * 60)
print("動画から球速を推定")
print("=" * 60)

print()
print("入力動画:")
print(INPUT_VIDEO)

print()
print("モデル:")
print(MODEL_PATH)


# ============================================================
# ファイル確認
# ============================================================

video_path = Path(INPUT_VIDEO)
model_path = Path(MODEL_PATH)

if not video_path.exists():
    raise FileNotFoundError(f"動画が見つかりません: {INPUT_VIDEO}")

if not model_path.exists():
    raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")


# ============================================================
# モデル読み込み
# ============================================================

print()
print("=" * 60)
print("モデル読み込み")
print("=" * 60)

model = joblib.load(model_path)

print("モデル読み込み完了")


# ============================================================
# ボール追跡
# ============================================================

print()
print("=" * 60)
print("ボール追跡")
print("=" * 60)

print()
print("追跡対象:")
print(INPUT_VIDEO)


# 既存のtrack_ball.pyを実行
result = subprocess.run(
    [
        sys.executable,
        "src/track_ball.py",
        str(video_path),
    ],
    capture_output=True,
    text=True,
)


print(result.stdout)

if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("ボール追跡に失敗しました")


# ============================================================
# 追跡CSV確認
# ============================================================

tracking_path = Path("output/ball_tracking.csv")

if not tracking_path.exists():
    raise FileNotFoundError("追跡CSVが作成されませんでした: " + str(tracking_path))

print()
print("追跡CSV:")
print(tracking_path)


# ============================================================
# CSV読み込み
# ============================================================

df = pd.read_csv(tracking_path)

print()
print("追跡データ数:", len(df))


# ============================================================
# 必要特徴量確認
# ============================================================

missing = [feature for feature in FEATURES if feature not in df.columns]

if missing:

    print()
    print("必要な特徴量がありません:")

    for feature in missing:
        print(" -", feature)

    print()
    print("現在のCSVに存在する列:")
    print(df.columns.tolist())

    raise ValueError("TOP_4特徴量を作成できません")


# ============================================================
# 特徴量取得
# ============================================================

X = df[FEATURES].copy()


# ============================================================
# 欠損値処理
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
# 結果
# ============================================================

predicted_speed = prediction[0]

print()
print("=" * 60)
print("推定結果")
print("=" * 60)

print()
print(f"推定球速: {predicted_speed:.2f} km/h")


# ============================================================
# 特徴量表示
# ============================================================

print()
print("使用した特徴量")
print("-" * 60)

for feature in FEATURES:
    print(f"{feature:25s}: " f"{X.iloc[0][feature]:.6f}")


# ============================================================
# 結果保存
# ============================================================

result_df = pd.DataFrame(
    {
        "video": [video_path.name],
        "predicted_speed_kmh": [predicted_speed],
        "std_dy": [X.iloc[0]["std_dy"]],
        "path_straightness": [X.iloc[0]["path_straightness"]],
        "std_pixel_speed": [X.iloc[0]["std_pixel_speed"]],
        "std_dx": [X.iloc[0]["std_dx"]],
    }
)


output_path = Path("output/predicted_speed_result.csv")

result_df.to_csv(
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
