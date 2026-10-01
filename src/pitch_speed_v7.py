import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# 投球速度解析 v7
#
# v5のボール追跡CSV
# +
# ピッチャーズプレートのホモグラフィ
#
# 画像座標 → プレート基準の実距離
# → 速度[m/s]
# → 速度[km/h]
#
# 注意：
# ボールは空中にあるため、地面平面のホモグラフィを
# そのまま適用した値は「暫定的な実距離・速度」です。
# ============================================================


# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")

CALIBRATION_FILE = Path("output/plate_calibration/pitcher_plate_calibration.npz")

OUTPUT_DIR = Path("output/pitch_speed_v7")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# confidenceが低すぎる点を速度計算から除外
MIN_CONFIDENCE = 0.3


# グラフ表示
SHOW_GRAPH = True


# ============================================================
# キャリブレーション読み込み
# ============================================================

print()
print("============================================================")
print("投球速度解析 v7")
print("ピッチャーズプレート基準キャリブレーション")
print("============================================================")

print()
print("入力:", INPUT_DIR)
print("キャリブレーション:", CALIBRATION_FILE)
print("出力:", OUTPUT_DIR)


if not CALIBRATION_FILE.exists():

    print()
    print("キャリブレーションファイルがありません:")
    print(CALIBRATION_FILE)

    raise SystemExit


calibration = np.load(CALIBRATION_FILE)

H = calibration["H"]

print()
print("ホモグラフィ行列:")
print(H)


# ============================================================
# 画像座標 → 実世界座標
# ============================================================


def image_to_world(x, y):

    point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(point, H)

    X = transformed[0, 0, 0]
    Y = transformed[0, 0, 1]

    return X, Y


# ============================================================
# CSV解析
# ============================================================


def analyze_csv(csv_path):

    print()
    print("============================================================")
    print("動画:", csv_path.stem)
    print("============================================================")

    df = pd.read_csv(csv_path)

    required = ["frame", "time_sec", "cx", "cy", "confidence", "interpolated"]

    missing = [c for c in required if c not in df.columns]

    if missing:

        print("必要な列がありません:", missing)

        return None

    print("追跡点数:", len(df))

    # ========================================================
    # confidenceフィルタ
    # ========================================================

    valid = df[df["confidence"] >= MIN_CONFIDENCE].copy()

    if len(valid) < 5:

        print("有効点が少なすぎます")

        return None

    print("confidence >= " f"{MIN_CONFIDENCE} の点:", len(valid))

    # ========================================================
    # 実世界座標へ変換
    # ========================================================

    world_x = []
    world_y = []

    for _, row in valid.iterrows():

        X, Y = image_to_world(row["cx"], row["cy"])

        world_x.append(X)

        world_y.append(Y)

    valid["world_x_m"] = world_x
    valid["world_y_m"] = world_y

    # ========================================================
    # 時間
    # ========================================================

    t = valid["time_sec"].values.astype(float)

    X = valid["world_x_m"].values.astype(float)

    Y = valid["world_y_m"].values.astype(float)

    # ========================================================
    # 隣接点間の距離
    # ========================================================

    dx = np.diff(X)
    dy = np.diff(Y)
    dt = np.diff(t)

    distance = np.sqrt(dx**2 + dy**2)

    # 0秒除外
    valid_interval = dt > 0

    distance = distance[valid_interval]

    dt_valid = dt[valid_interval]

    if len(distance) == 0:

        print("有効な速度区間がありません")

        return None

    speed_ms = distance / dt_valid

    speed_kmh = speed_ms * 3.6

    # ========================================================
    # 結果
    # ========================================================

    average_speed_ms = np.mean(speed_ms)

    average_speed_kmh = average_speed_ms * 3.6

    maximum_speed_ms = np.max(speed_ms)

    maximum_speed_kmh = maximum_speed_ms * 3.6

    max_index = np.argmax(speed_kmh)

    # 元のframe番号
    valid_frames = valid["frame"].values

    # 区間の開始フレーム
    valid_frame_indices = np.where(valid_interval)[0]

    max_interval_index = valid_frame_indices[max_index]

    frame_start = int(valid_frames[max_interval_index])

    frame_end = int(valid_frames[max_interval_index + 1])

    # ========================================================
    # 総移動距離
    # ========================================================

    total_distance = np.sum(distance)

    # ========================================================
    # 表示
    # ========================================================

    print()
    print("------------------------------------------")
    print("実距離解析")
    print("------------------------------------------")

    print(f"総移動距離      : " f"{total_distance:.3f} m")

    print(f"平均速度        : " f"{average_speed_ms:.3f} m/s")

    print(f"平均速度        : " f"{average_speed_kmh:.2f} km/h")

    print(f"最大瞬間速度    : " f"{maximum_speed_ms:.3f} m/s")

    print(f"最大瞬間速度    : " f"{maximum_speed_kmh:.2f} km/h")

    print(f"最大速度区間    : " f"Frame {frame_start} → {frame_end}")

    # ========================================================
    # 各点の速度
    # ========================================================

    point_speed_ms = np.full(len(valid), np.nan)

    point_speed_kmh = np.full(len(valid), np.nan)

    interval_indices = np.where(valid_interval)[0]

    for i, interval_index in enumerate(interval_indices):

        point_speed_ms[interval_index] = speed_ms[i]

        point_speed_kmh[interval_index] = speed_kmh[i]

    valid["speed_ms"] = point_speed_ms

    valid["speed_kmh"] = point_speed_kmh

    # ========================================================
    # CSV保存
    # ========================================================

    output_csv = OUTPUT_DIR / f"{csv_path.stem}_v7.csv"

    valid.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ========================================================
    # グラフ
    # ========================================================

    output_png = OUTPUT_DIR / f"{csv_path.stem}_v7.png"

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --------------------------------------------------------
    # 実世界座標での軌跡
    # --------------------------------------------------------

    axes[0].plot(X, Y, marker="o")

    axes[0].set_xlabel("World X [m]")

    axes[0].set_ylabel("World Y [m]")

    axes[0].set_title(csv_path.stem + " - Calibrated trajectory")

    axes[0].grid(True, alpha=0.3)

    # --------------------------------------------------------
    # 速度
    # --------------------------------------------------------

    speed_times = t[:-1][valid_interval]

    axes[1].plot(speed_times, speed_kmh, marker="o")

    axes[1].set_xlabel("Time [s]")

    axes[1].set_ylabel("Speed [km/h]")

    axes[1].set_title("Estimated speed")

    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_png, dpi=200)

    if SHOW_GRAPH:

        plt.show()

    plt.close()

    print()
    print("CSV:", output_csv)

    print("グラフ:", output_png)

    return {
        "name": csv_path.stem,
        "points": len(valid),
        "distance": total_distance,
        "average_kmh": average_speed_kmh,
        "maximum_kmh": maximum_speed_kmh,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


# ============================================================
# 全CSV解析
# ============================================================

csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))


if not csv_files:

    print()
    print("追跡CSVが見つかりません")

    raise SystemExit


results = []


for csv_path in csv_files:

    result = analyze_csv(csv_path)

    if result is not None:

        results.append(result)


# ============================================================
# 最終結果
# ============================================================

print()
print("============================================================")
print("全動画の速度解析完了")
print("============================================================")

print()


for r in results:

    print(
        f"{r['name']} | "
        f"平均: {r['average_kmh']:.2f} km/h | "
        f"最大: {r['maximum_kmh']:.2f} km/h | "
        f"距離: {r['distance']:.3f} m"
    )


print()
print("出力先:", OUTPUT_DIR)
