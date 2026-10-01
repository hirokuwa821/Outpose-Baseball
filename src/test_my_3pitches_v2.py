from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 設定
# ============================================================

MODEL_PATH = Path("output/train839_feature_selection/best_train839_speed_model.pkl")

TRACKING_DIR = Path("output/my_3pitches_test")

OUTPUT_DIR = Path("output/my_3pitches_test_v2")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# あなたの3投球
# ------------------------------------------------------------

TEST_DATA = [
    {
        "tracking_csv": TRACKING_DIR / "pitch_1_tracking.csv",
        "video": "pitch_102_1.mp4",
        "actual_speed": 102.0,
    },
    {
        "tracking_csv": TRACKING_DIR / "pitch_2_tracking.csv",
        "video": "pitch_98_1.mp4",
        "actual_speed": 98.0,
    },
    {
        "tracking_csv": TRACKING_DIR / "pitch_3_tracking.csv",
        "video": "pitch_98_2.mp4",
        "actual_speed": 98.0,
    },
]


# ============================================================
# TOP_20特徴量
# ============================================================

FEATURES = [
    "start_x",
    "std_dy",
    "path_straightness",
    "median_step_distance",
    "max_pixel_speed",
    "average_pixel_speed",
    "frame_start",
    "mean_step_distance",
    "frame_end",
    "end_x",
    "x_range",
    "std_pixel_speed",
    "start_end_distance_px",
    "total_distance_px",
    "start_y",
    "total_dx",
    "max_step_distance",
    "mean_dx",
    "std_dx",
    "trajectory_angle_deg",
]


# ============================================================
# 表示
# ============================================================

print("=" * 60)
print("自分の3投球 TOP_20モデル検証 v2")
print("=" * 60)


# ============================================================
# モデル読み込み
# ============================================================

print()
print("モデル:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"モデルがありません: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# モデルの特徴量名確認
# ============================================================

print()
print("=" * 60)
print("モデル特徴量確認")
print("=" * 60)

if hasattr(model, "feature_names_in_"):

    model_features = list(model.feature_names_in_)

    print()
    print("モデルが要求する特徴量:")

    for i, feature in enumerate(model_features, start=1):
        print(f"{i:2d}. {feature}")

else:

    model_features = FEATURES

    print("feature_names_in_ がないため、" "設定したTOP_20を使用します。")


# ============================================================
# 特徴量作成
# ============================================================


def make_features(csv_path):

    print()
    print("-" * 60)
    print("特徴量作成")
    print("-" * 60)

    print("入力CSV:", csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(f"追跡CSVがありません: {csv_path}")

    df = pd.read_csv(csv_path)

    print("追跡点数:", len(df))

    print("列:", list(df.columns))

    # --------------------------------------------------------
    # 座標列
    # --------------------------------------------------------

    if "center_x" not in df.columns:
        raise ValueError("center_x がありません")

    if "center_y" not in df.columns:
        raise ValueError("center_y がありません")

    x = pd.to_numeric(df["center_x"], errors="coerce").to_numpy(dtype=float)

    y = pd.to_numeric(df["center_y"], errors="coerce").to_numpy(dtype=float)

    # --------------------------------------------------------
    # フレーム
    # --------------------------------------------------------

    frame = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(frame)

    x = x[valid]
    y = y[valid]
    frame = frame[valid]

    if len(x) < 3:

        raise ValueError(f"有効な追跡点が少なすぎます: {len(x)}")

    # --------------------------------------------------------
    # フレーム順に並べる
    # --------------------------------------------------------

    order = np.argsort(frame)

    x = x[order]
    y = y[order]
    frame = frame[order]

    # --------------------------------------------------------
    # 差分
    # --------------------------------------------------------

    dx = np.diff(x)
    dy = np.diff(y)

    step_distance = np.sqrt(dx**2 + dy**2)

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if len(frame) >= 2:

        frame_diff = np.diff(frame)

        frame_diff = frame_diff[frame_diff > 0]

        if len(frame_diff) > 0:

            # CSVのフレーム間隔からFPS推定
            fps = 30.0

        else:

            fps = 30.0

    else:

        fps = 30.0

    # --------------------------------------------------------
    # 基本値
    # --------------------------------------------------------

    frame_start = frame[0]
    frame_end = frame[-1]

    num_frames = len(frame)

    total_dx = x[-1] - x[0]
    total_dy = y[-1] - y[0]

    total_distance_px = np.sum(step_distance)

    start_end_distance_px = np.sqrt(total_dx**2 + total_dy**2)

    duration_sec = (frame_end - frame_start) / fps

    # --------------------------------------------------------
    # pixel speed
    # --------------------------------------------------------

    pixel_speed = step_distance * fps

    average_pixel_speed = np.mean(pixel_speed)

    max_pixel_speed = np.max(pixel_speed)

    std_pixel_speed = np.std(pixel_speed)

    # --------------------------------------------------------
    # step distance
    # --------------------------------------------------------

    mean_step_distance = np.mean(step_distance)

    std_step_distance = np.std(step_distance)

    max_step_distance = np.max(step_distance)

    median_step_distance = np.median(step_distance)

    # --------------------------------------------------------
    # path straightness
    # --------------------------------------------------------

    if total_distance_px > 0:

        path_straightness = start_end_distance_px / total_distance_px

    else:

        path_straightness = 0.0

    # --------------------------------------------------------
    # trajectory angle
    # --------------------------------------------------------

    trajectory_angle_deg = np.degrees(np.arctan2(total_dy, total_dx))

    # --------------------------------------------------------
    # mean / std
    # --------------------------------------------------------

    mean_dx = np.mean(dx)
    mean_dy = np.mean(dy)

    std_dx = np.std(dx)
    std_dy = np.std(dy)

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    features = {
        "start_x": x[0],
        "std_dy": std_dy,
        "path_straightness": path_straightness,
        "median_step_distance": median_step_distance,
        "max_pixel_speed": max_pixel_speed,
        "average_pixel_speed": average_pixel_speed,
        "frame_start": frame_start,
        "mean_step_distance": mean_step_distance,
        "frame_end": frame_end,
        "end_x": x[-1],
        "x_range": np.max(x) - np.min(x),
        "std_pixel_speed": std_pixel_speed,
        "start_end_distance_px": start_end_distance_px,
        "total_distance_px": total_distance_px,
        "start_y": y[0],
        "total_dx": total_dx,
        "max_step_distance": max_step_distance,
        "mean_dx": mean_dx,
        "std_dx": std_dx,
        "trajectory_angle_deg": trajectory_angle_deg,
    }

    return pd.DataFrame([features])


# ============================================================
# 3本を処理
# ============================================================

all_results = []
all_features = []


for i, item in enumerate(TEST_DATA, start=1):

    print()
    print()
    print("#" * 60)
    print(f"投球 {i}/3")
    print("#" * 60)

    csv_path = item["tracking_csv"]

    video_name = item["video"]

    actual_speed = item["actual_speed"]

    print()
    print("動画:", video_name)

    print("実測:", actual_speed, "km/h")

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    feature_df = make_features(csv_path)

    # --------------------------------------------------------
    # 特徴量確認
    # --------------------------------------------------------

    missing = [f for f in FEATURES if f not in feature_df.columns]

    if missing:

        raise ValueError("必要特徴量がありません:\n" + "\n".join(missing))

    # --------------------------------------------------------
    # TOP_20
    # --------------------------------------------------------

    X = feature_df[FEATURES].copy()

    # --------------------------------------------------------
    # モデル要求順に並べる
    # --------------------------------------------------------

    if set(model_features) == set(X.columns):

        X = X[model_features]

    else:

        print()
        print("注意: モデルの特徴量と" "計算した特徴量が一致しません。")

        print("モデル:", model_features)

        print("計算:", list(X.columns))

        raise ValueError("モデルと特徴量が一致しません")

    # --------------------------------------------------------
    # 特徴量表示
    # --------------------------------------------------------

    print()
    print("入力特徴量:")

    for feature in X.columns:

        print(f"{feature:28s}" f": {X.iloc[0][feature]:.6f}")

    # --------------------------------------------------------
    # 予測
    # --------------------------------------------------------

    prediction = float(model.predict(X)[0])

    error = prediction - actual_speed

    absolute_error = abs(error)

    # --------------------------------------------------------
    # 結果
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("推定結果")
    print("=" * 60)

    print(f"実測球速 : " f"{actual_speed:.2f} km/h")

    print(f"推定球速 : " f"{prediction:.2f} km/h")

    print(f"誤差     : " f"{error:+.2f} km/h")

    print(f"絶対誤差 : " f"{absolute_error:.2f} km/h")

    # --------------------------------------------------------
    # 保存用
    # --------------------------------------------------------

    row = {
        "video": video_name,
        "tracking_csv": str(csv_path),
        "actual_speed_kmh": actual_speed,
        "predicted_speed_kmh": prediction,
        "error_kmh": error,
        "absolute_error_kmh": absolute_error,
    }

    for feature in FEATURES:

        row[feature] = float(feature_df.iloc[0][feature])

    all_results.append(row)

    feature_row = {"video": video_name}

    for feature in FEATURES:

        feature_row[feature] = float(feature_df.iloc[0][feature])

    all_features.append(feature_row)


# ============================================================
# 結果DataFrame
# ============================================================

result_df = pd.DataFrame(all_results)

feature_df = pd.DataFrame(all_features)


# ============================================================
# 評価
# ============================================================

mae = np.mean(result_df["absolute_error_kmh"])

rmse = np.sqrt(np.mean(result_df["error_kmh"] ** 2))

mean_error = np.mean(result_df["error_kmh"])


# ============================================================
# 保存
# ============================================================

prediction_file = OUTPUT_DIR / "my_3pitches_predictions_v2.csv"

feature_file = OUTPUT_DIR / "my_3pitches_features_v2.csv"

result_df.to_csv(prediction_file, index=False, encoding="utf-8-sig")

feature_df.to_csv(feature_file, index=False, encoding="utf-8-sig")


# ============================================================
# 最終結果
# ============================================================

print()
print()
print("=" * 60)
print("3本の未知データ検証結果")
print("=" * 60)

print()

for _, row in result_df.iterrows():

    print(f"{row['video']}")

    print(f"  実測 : " f"{row['actual_speed_kmh']:.2f} km/h")

    print(f"  推定 : " f"{row['predicted_speed_kmh']:.2f} km/h")

    print(f"  誤差 : " f"{row['error_kmh']:+.2f} km/h")

    print(f"  |誤差|: " f"{row['absolute_error_kmh']:.2f} km/h")

    print()


print("=" * 60)
print("3本総合")
print("=" * 60)

print()
print(f"MAE      : " f"{mae:.3f} km/h")

print(f"RMSE     : " f"{rmse:.3f} km/h")

print(f"平均誤差 : " f"{mean_error:+.3f} km/h")

print()

print("予測結果:")
print(prediction_file)

print()
print("特徴量:")
print(feature_file)

print()
print("=" * 60)
print("検証完了")
print("=" * 60)
