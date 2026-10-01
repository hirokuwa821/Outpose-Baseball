import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 投球速度解析 v8
#
# 目的：
# v5で得られたボール追跡から、
# 「実際の投球に使えそうな区間」を抽出する。
#
# 今回はまだ最終球速を確定しない。
#
# 出力：
# ・追跡開始フレーム
# ・追跡終了フレーム
# ・飛行時間候補
# ・18.44 mを仮定した平均速度
# ・候補区間CSV
# ============================================================


INPUT_DIR = Path("output/ball_tracking_v5")

OUTPUT_DIR = Path("output/pitch_speed_v8")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 投手板からホームプレートまでの規定距離
# ------------------------------------------------------------

PITCHING_DISTANCE_M = 18.44


# ------------------------------------------------------------
# 最低confidence
# ------------------------------------------------------------

MIN_CONFIDENCE = 0.3


# ------------------------------------------------------------
# 最低移動距離
#
# あまり動いていない誤検出区間を除外する。
# ------------------------------------------------------------

MIN_DISTANCE_PX = 30.0


# ============================================================
# CSV読み込み
# ============================================================


def analyze(csv_path):

    print()
    print("============================================================")
    print("動画:", csv_path.stem)
    print("============================================================")

    df = pd.read_csv(csv_path)

    required_columns = ["frame", "time_sec", "cx", "cy", "confidence", "interpolated"]

    for column in required_columns:

        if column not in df.columns:

            print("必要な列がありません:", column)

            return None

    print("追跡点数:", len(df))

    # ========================================================
    # confidenceフィルタ
    # ========================================================

    valid = df[df["confidence"] >= MIN_CONFIDENCE].copy()

    if len(valid) < 5:

        print("有効追跡点が少なすぎます")

        return None

    valid = valid.sort_values("frame").reset_index(drop=True)

    # ========================================================
    # 座標
    # ========================================================

    x = valid["cx"].values
    y = valid["cy"].values

    frames = valid["frame"].values
    times = valid["time_sec"].values

    # ========================================================
    # 追跡区間全体
    # ========================================================

    start_frame = int(frames[0])

    end_frame = int(frames[-1])

    start_time = float(times[0])

    end_time = float(times[-1])

    total_time = end_time - start_time

    # ========================================================
    # 画像上の総移動距離
    # ========================================================

    dx = np.diff(x)
    dy = np.diff(y)

    step_distance = np.sqrt(dx**2 + dy**2)

    total_distance_px = np.sum(step_distance)

    direct_distance_px = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

    # ========================================================
    # 画像上の移動方向
    # ========================================================

    direction_x = x[-1] - x[0]

    direction_y = y[-1] - y[0]

    # ========================================================
    # 速度
    # ========================================================

    if total_time > 0:

        average_px_s = total_distance_px / total_time

    else:

        average_px_s = 0

    # ========================================================
    # 規定距離を仮定した場合の平均速度
    #
    # 注意：
    # これは「飛行時間が本当に投球区間だった場合」の
    # 仮定値。
    # ========================================================

    if total_time > 0:

        assumed_speed_ms = PITCHING_DISTANCE_M / total_time

        assumed_speed_kmh = assumed_speed_ms * 3.6

    else:

        assumed_speed_kmh = 0

    # ========================================================
    # 結果表示
    # ========================================================

    print()
    print("------------------------------------------")
    print("追跡区間")
    print("------------------------------------------")

    print(f"Frame          : " f"{start_frame} → {end_frame}")

    print(f"時間           : " f"{total_time:.4f} sec")

    print(f"フレーム数     : " f"{end_frame - start_frame + 1}")

    print(f"画像上移動距離 : " f"{total_distance_px:.2f} px")

    print(f"直線距離       : " f"{direct_distance_px:.2f} px")

    print(f"平均画像速度   : " f"{average_px_s:.2f} px/s")

    print()
    print("------------------------------------------")
    print("18.44 mを仮定した場合")
    print("------------------------------------------")

    print(f"平均速度       : " f"{assumed_speed_kmh:.2f} km/h")

    print("※この値は投球区間が正しく特定できた場合のみ有効")

    print()
    print("------------------------------------------")
    print("移動方向")
    print("------------------------------------------")

    print(f"dx : {direction_x:.2f} px")

    print(f"dy : {direction_y:.2f} px")

    # ========================================================
    # 各フレーム間の情報
    # ========================================================

    result = valid.copy()

    result["step_dx"] = np.nan
    result["step_dy"] = np.nan
    result["step_distance_px"] = np.nan
    result["step_time_sec"] = np.nan
    result["step_speed_px_s"] = np.nan

    for i in range(1, len(result)):

        dx_i = x[i] - x[i - 1]

        dy_i = y[i] - y[i - 1]

        distance_i = np.sqrt(dx_i**2 + dy_i**2)

        dt_i = times[i] - times[i - 1]

        if dt_i > 0:

            speed_i = distance_i / dt_i

        else:

            speed_i = np.nan

        result.loc[i, "step_dx"] = dx_i

        result.loc[i, "step_dy"] = dy_i

        result.loc[i, "step_distance_px"] = distance_i

        result.loc[i, "step_time_sec"] = dt_i

        result.loc[i, "step_speed_px_s"] = speed_i

    # ========================================================
    # CSV保存
    # ========================================================

    output_csv = OUTPUT_DIR / f"{csv_path.stem}_v8.csv"

    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print()
    print("CSV:", output_csv)

    return {
        "name": csv_path.stem,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "time": total_time,
        "distance_px": total_distance_px,
        "assumed_kmh": assumed_speed_kmh,
    }


# ============================================================
# 全動画
# ============================================================

csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))


if not csv_files:

    print()
    print("追跡CSVが見つかりません")

    print("確認:", INPUT_DIR)

    raise SystemExit


results = []


for csv_path in csv_files:

    result = analyze(csv_path)

    if result is not None:

        results.append(result)


# ============================================================
# 最終結果
# ============================================================

print()
print("============================================================")
print("v8解析完了")
print("============================================================")

print()

for r in results:

    print(
        f"{r['name']} | "
        f"Frame {r['start_frame']} → "
        f"{r['end_frame']} | "
        f"時間 {r['time']:.4f} sec | "
        f"仮定速度 {r['assumed_kmh']:.2f} km/h"
    )


print()
print("出力先:", OUTPUT_DIR)
