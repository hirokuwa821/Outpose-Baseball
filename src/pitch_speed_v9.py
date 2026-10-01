import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 投球速度解析 v9
#
# 目的：
# v5の追跡軌跡から、投球区間の候補を複数抽出する。
#
# 今回は球速を確定しない。
#
# ・連続した追跡区間
# ・移動方向
# ・移動距離
# ・速度
# ・飛行時間
# ・18.44 mを仮定した速度
#
# を候補ごとに表示する。
# ============================================================


INPUT_DIR = Path("output/ball_tracking_v5")

OUTPUT_DIR = Path("output/pitch_speed_v9")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 設定
# ============================================================

PITCHING_DISTANCE_M = 18.44

MIN_CONFIDENCE = 0.3

# 候補区間の最低フレーム数
MIN_POINTS = 8

# 速度計算時の異常値除外
MAX_SPEED_MULTIPLIER = 3.0


# ============================================================
# 解析
# ============================================================


def analyze(csv_path):

    print()
    print("=" * 60)
    print("動画:", csv_path.stem)
    print("=" * 60)

    df = pd.read_csv(csv_path)

    required = [
        "frame",
        "time_sec",
        "cx",
        "cy",
        "confidence",
    ]

    for col in required:

        if col not in df.columns:

            print("必要な列がありません:", col)

            return

    # --------------------------------------------------------
    # confidenceフィルタ
    # --------------------------------------------------------

    df = df[df["confidence"] >= MIN_CONFIDENCE].copy()

    df = df.sort_values("frame").reset_index(drop=True)

    if len(df) < MIN_POINTS:

        print("有効追跡点が不足しています")

        return

    print("有効追跡点数:", len(df))

    # ========================================================
    # 各フレーム間の移動量を計算
    # ========================================================

    df["dx"] = df["cx"].diff()

    df["dy"] = df["cy"].diff()

    df["distance_px"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    df["dt"] = df["time_sec"].diff()

    df["speed_px_s"] = df["distance_px"] / df["dt"]

    # ========================================================
    # 速度の中央値
    # ========================================================

    speed_values = df["speed_px_s"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(speed_values) == 0:

        print("速度を計算できません")

        return

    median_speed = speed_values.median()

    # ========================================================
    # 異常な速度を除外
    #
    # 極端な誤検出によるジャンプを除外する。
    # ========================================================

    speed_limit = median_speed * MAX_SPEED_MULTIPLIER

    df["valid_motion"] = df["speed_px_s"] <= speed_limit

    # 最初の点は速度がない
    df.loc[df.index[0], "valid_motion"] = False

    # ========================================================
    # 連続区間を作る
    # ========================================================

    segments = []

    current = []

    for i in range(len(df)):

        row = df.iloc[i]

        if row["valid_motion"]:

            current.append(i)

        else:

            if len(current) >= MIN_POINTS:

                segments.append(current.copy())

            current = []

    if len(current) >= MIN_POINTS:

        segments.append(current.copy())

    # ========================================================
    # 区間が見つからない場合
    # ========================================================

    if not segments:

        print("連続した投球候補区間が見つかりません")

        print("全追跡区間を候補として解析します。")

        segments = [list(range(1, len(df)))]

    print()
    print("------------------------------------------")
    print("投球区間候補")
    print("------------------------------------------")

    candidates = []

    # ========================================================
    # 各候補区間を解析
    # ========================================================

    for number, indices in enumerate(segments, start=1):

        part = df.iloc[indices].copy()

        if len(part) < MIN_POINTS:

            continue

        start_frame = int(part["frame"].iloc[0])

        end_frame = int(part["frame"].iloc[-1])

        start_time = float(part["time_sec"].iloc[0])

        end_time = float(part["time_sec"].iloc[-1])

        flight_time = end_time - start_time

        if flight_time <= 0:

            continue

        # ----------------------------------------------------
        # 移動距離
        # ----------------------------------------------------

        total_distance_px = part["distance_px"].sum()

        direct_distance_px = np.sqrt(
            (part["cx"].iloc[-1] - part["cx"].iloc[0]) ** 2
            + (part["cy"].iloc[-1] - part["cy"].iloc[0]) ** 2
        )

        # ----------------------------------------------------
        # 平均画像速度
        # ----------------------------------------------------

        average_px_s = total_distance_px / flight_time

        # ----------------------------------------------------
        # 最大画像速度
        # ----------------------------------------------------

        max_px_s = part["speed_px_s"].max()

        # ----------------------------------------------------
        # 18.44m仮定
        # ----------------------------------------------------

        assumed_speed_ms = PITCHING_DISTANCE_M / flight_time

        assumed_speed_kmh = assumed_speed_ms * 3.6

        # ----------------------------------------------------
        # 移動方向
        # ----------------------------------------------------

        dx_total = part["cx"].iloc[-1] - part["cx"].iloc[0]

        dy_total = part["cy"].iloc[-1] - part["cy"].iloc[0]

        angle = np.degrees(np.arctan2(dy_total, dx_total))

        candidate = {
            "candidate": number,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "points": len(part),
            "flight_time_sec": flight_time,
            "distance_px": total_distance_px,
            "direct_distance_px": direct_distance_px,
            "average_px_s": average_px_s,
            "max_px_s": max_px_s,
            "assumed_speed_kmh": assumed_speed_kmh,
            "dx": dx_total,
            "dy": dy_total,
            "angle_deg": angle,
        }

        candidates.append(candidate)

        # ----------------------------------------------------
        # 表示
        # ----------------------------------------------------

        print()

        print(f"[候補 {number}]")

        print(f"Frame        : " f"{start_frame} → {end_frame}")

        print(f"追跡点数     : " f"{len(part)}")

        print(f"飛行時間     : " f"{flight_time:.4f} sec")

        print(f"移動距離     : " f"{total_distance_px:.2f} px")

        print(f"直線距離     : " f"{direct_distance_px:.2f} px")

        print(f"平均速度     : " f"{average_px_s:.2f} px/s")

        print(f"最大速度     : " f"{max_px_s:.2f} px/s")

        print(f"移動方向     : " f"{angle:.2f} deg")

        print(f"18.44m仮定   : " f"{assumed_speed_kmh:.2f} km/h")

    # ========================================================
    # CSV保存
    # ========================================================

    candidate_df = pd.DataFrame(candidates)

    output_csv = OUTPUT_DIR / f"{csv_path.stem}_v9_candidates.csv"

    candidate_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print()
    print("候補CSV:", output_csv)


# ============================================================
# 全動画
# ============================================================

csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))


if not csv_files:

    print("追跡CSVが見つかりません:")

    print(INPUT_DIR)

    raise SystemExit


for csv_path in csv_files:

    analyze(csv_path)


print()
print("=" * 60)
print("v9解析完了")
print("=" * 60)

print("出力先:", OUTPUT_DIR)
