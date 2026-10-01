import pandas as pd
import numpy as np


def detect_hand(csv_path="output/keypoints.csv"):

    df = pd.read_csv(csv_path)

    required = [
        "x_5",
        "y_5",
        "x_6",
        "y_6",
        "x_7",
        "y_7",
        "x_8",
        "y_8",
        "x_9",
        "y_9",
        "x_10",
        "y_10",
    ]

    for col in required:
        if col not in df.columns:
            print(f"列がありません: {col}")
            return "unknown"

    if len(df) < 20:
        print("フレーム数が少なすぎます")
        return "unknown"

    # -----------------------------------------
    # 肩の中心
    # -----------------------------------------

    shoulder_x = (df["x_5"] + df["x_6"]) / 2
    shoulder_y = (df["y_5"] + df["y_6"]) / 2

    # -----------------------------------------
    # 距離計算
    # -----------------------------------------

    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 手首 → 肩中心
    left_wrist_dist = distance(df["x_9"], df["y_9"], shoulder_x, shoulder_y)

    right_wrist_dist = distance(df["x_10"], df["y_10"], shoulder_x, shoulder_y)

    # 肘 → 肩中心
    left_elbow_dist = distance(df["x_7"], df["y_7"], shoulder_x, shoulder_y)

    right_elbow_dist = distance(df["x_8"], df["y_8"], shoulder_x, shoulder_y)

    # -----------------------------------------
    # 投球動作が含まれやすい後半50%を見る
    # -----------------------------------------

    start = int(len(df) * 0.50)

    left_wrist_dist = left_wrist_dist.iloc[start:]
    right_wrist_dist = right_wrist_dist.iloc[start:]

    left_elbow_dist = left_elbow_dist.iloc[start:]
    right_elbow_dist = right_elbow_dist.iloc[start:]

    # -----------------------------------------
    # 大きく伸びた上位10フレームの平均
    # -----------------------------------------

    def top_mean(series, n=10):

        values = series.replace([np.inf, -np.inf], np.nan).dropna().values

        if len(values) == 0:
            return 0.0

        values = np.sort(values)

        n = min(n, len(values))

        return float(np.mean(values[-n:]))

    left_wrist = top_mean(left_wrist_dist)
    right_wrist = top_mean(right_wrist_dist)

    left_elbow = top_mean(left_elbow_dist)
    right_elbow = top_mean(right_elbow_dist)

    # -----------------------------------------
    # 総合スコア
    #
    # 手首を主に使用
    # 肘は補助
    # -----------------------------------------

    left_score = left_wrist + 0.5 * left_elbow

    right_score = right_wrist + 0.5 * right_elbow

    print(f"左手首距離 : {left_wrist:.4f}")
    print(f"右手首距離 : {right_wrist:.4f}")

    print(f"左肘距離   : {left_elbow:.4f}")
    print(f"右肘距離   : {right_elbow:.4f}")

    print(f"左腕スコア : {left_score:.4f}")
    print(f"右腕スコア : {right_score:.4f}")

    # -----------------------------------------
    # 判定
    # -----------------------------------------

    if left_score <= 0 or right_score <= 0:
        return "unknown"

    if right_score > left_score:

        ratio = right_score / left_score

        print(f"右/左 比率 : {ratio:.3f}")

        if ratio >= 1.12:
            return "right"

    else:

        ratio = left_score / right_score

        print(f"左/右 比率 : {ratio:.3f}")

        if ratio >= 1.12:
            return "left"

    return "unknown"
