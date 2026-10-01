from pathlib import Path
import pandas as pd
import numpy as np

from pose import analyze_video

# ============================================================
# 設定
# ============================================================

VIDEO_DIR = Path("test_videos")
LABEL_FILE = Path("hand_labels.csv")
OUTPUT_FILE = Path("output/hand_dataset.csv")


# ============================================================
# YOLO Poseのキーポイント取得
# ============================================================


def extract_keypoints(results):

    rows = []

    for result in results:

        if result.keypoints is None:
            continue

        if result.keypoints.xy is None:
            continue

        xy = result.keypoints.xy.cpu().numpy()

        if len(xy) == 0:
            continue

        # 最初の人物だけ使用
        person = xy[0]

        # 17点のうち必要な12点
        required_indices = [
            5,  # 左肩
            6,  # 右肩
            7,  # 左肘
            8,  # 右肘
            9,  # 左手首
            10,  # 右手首
            11,  # 左腰
            12,  # 右腰
        ]

        values = []

        for idx in required_indices:

            if idx >= len(person):
                values.extend([np.nan, np.nan])
            else:
                values.extend([person[idx][0], person[idx][1]])

        rows.append(values)

    columns = [
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
        "x_11",
        "y_11",
        "x_12",
        "y_12",
    ]

    return pd.DataFrame(rows, columns=columns)


# ============================================================
# 補間
# ============================================================


def interpolate(df):

    df = df.copy()

    for col in df.columns:

        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
        )

    return df


# ============================================================
# 角度
# ============================================================


def angle(a, b, c):

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)

    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.degrees(np.arccos(cos_value))


# ============================================================
# 動画1本から特徴量を作成
# ============================================================


def create_hand_features(df):

    if len(df) < 20:
        return None

    df = interpolate(df)

    # --------------------------------------------------------
    # 肩中心
    # --------------------------------------------------------

    shoulder_x = (df["x_5"] + df["x_6"]) / 2
    shoulder_y = (df["y_5"] + df["y_6"]) / 2

    shoulder = np.column_stack([shoulder_x, shoulder_y])

    # --------------------------------------------------------
    # 各部位
    # --------------------------------------------------------

    left_elbow = np.column_stack([df["x_7"], df["y_7"]])

    right_elbow = np.column_stack([df["x_8"], df["y_8"]])

    left_wrist = np.column_stack([df["x_9"], df["y_9"]])

    right_wrist = np.column_stack([df["x_10"], df["y_10"]])

    # --------------------------------------------------------
    # 肩からの距離
    # --------------------------------------------------------

    left_wrist_dist = np.linalg.norm(left_wrist - shoulder, axis=1)

    right_wrist_dist = np.linalg.norm(right_wrist - shoulder, axis=1)

    left_elbow_dist = np.linalg.norm(left_elbow - shoulder, axis=1)

    right_elbow_dist = np.linalg.norm(right_elbow - shoulder, axis=1)

    # --------------------------------------------------------
    # フレーム間速度
    # --------------------------------------------------------

    left_wrist_speed = np.linalg.norm(np.diff(left_wrist, axis=0), axis=1)

    right_wrist_speed = np.linalg.norm(np.diff(right_wrist, axis=0), axis=1)

    left_elbow_speed = np.linalg.norm(np.diff(left_elbow, axis=0), axis=1)

    right_elbow_speed = np.linalg.norm(np.diff(right_elbow, axis=0), axis=1)

    # --------------------------------------------------------
    # 後半部分
    # --------------------------------------------------------

    start = int(len(df) * 0.5)

    lw_dist = left_wrist_dist[start:]
    rw_dist = right_wrist_dist[start:]

    le_dist = left_elbow_dist[start:]
    re_dist = right_elbow_dist[start:]

    lw_speed = left_wrist_speed[max(0, start - 1) :]
    rw_speed = right_wrist_speed[max(0, start - 1) :]

    le_speed = left_elbow_speed[max(0, start - 1) :]
    re_speed = right_elbow_speed[max(0, start - 1) :]

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    def safe_mean(x):
        return float(np.nanmean(x))

    def safe_std(x):
        return float(np.nanstd(x))

    def safe_max(x):
        return float(np.nanmax(x))

    def safe_p90(x):
        return float(np.nanpercentile(x, 90))

    features = {
        # フレーム数
        "frame_count": len(df),
        # 左手首
        "left_wrist_dist_mean": safe_mean(lw_dist),
        "left_wrist_dist_std": safe_std(lw_dist),
        "left_wrist_dist_max": safe_max(lw_dist),
        "left_wrist_dist_p90": safe_p90(lw_dist),
        "left_wrist_speed_mean": safe_mean(lw_speed),
        "left_wrist_speed_std": safe_std(lw_speed),
        "left_wrist_speed_max": safe_max(lw_speed),
        "left_wrist_speed_p90": safe_p90(lw_speed),
        # 右手首
        "right_wrist_dist_mean": safe_mean(rw_dist),
        "right_wrist_dist_std": safe_std(rw_dist),
        "right_wrist_dist_max": safe_max(rw_dist),
        "right_wrist_dist_p90": safe_p90(rw_dist),
        "right_wrist_speed_mean": safe_mean(rw_speed),
        "right_wrist_speed_std": safe_std(rw_speed),
        "right_wrist_speed_max": safe_max(rw_speed),
        "right_wrist_speed_p90": safe_p90(rw_speed),
        # 左肘
        "left_elbow_dist_mean": safe_mean(le_dist),
        "left_elbow_dist_std": safe_std(le_dist),
        "left_elbow_dist_max": safe_max(le_dist),
        "left_elbow_dist_p90": safe_p90(le_dist),
        "left_elbow_speed_mean": safe_mean(le_speed),
        "left_elbow_speed_std": safe_std(le_speed),
        "left_elbow_speed_max": safe_max(le_speed),
        "left_elbow_speed_p90": safe_p90(le_speed),
        # 右肘
        "right_elbow_dist_mean": safe_mean(re_dist),
        "right_elbow_dist_std": safe_std(re_dist),
        "right_elbow_dist_max": safe_max(re_dist),
        "right_elbow_dist_p90": safe_p90(re_dist),
        "right_elbow_speed_mean": safe_mean(re_speed),
        "right_elbow_speed_std": safe_std(re_speed),
        "right_elbow_speed_max": safe_max(re_speed),
        "right_elbow_speed_p90": safe_p90(re_speed),
    }

    # --------------------------------------------------------
    # 左右差
    # --------------------------------------------------------

    features["wrist_dist_diff"] = (
        features["left_wrist_dist_mean"] - features["right_wrist_dist_mean"]
    )

    features["wrist_speed_diff"] = (
        features["left_wrist_speed_mean"] - features["right_wrist_speed_mean"]
    )

    features["elbow_dist_diff"] = (
        features["left_elbow_dist_mean"] - features["right_elbow_dist_mean"]
    )

    features["elbow_speed_diff"] = (
        features["left_elbow_speed_mean"] - features["right_elbow_speed_mean"]
    )

    return features


# ============================================================
# メイン
# ============================================================


def main():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 動画
    # --------------------------------------------------------

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    # 手動ラベル付きだけ
    videos = [
        v for v in videos if v.stem.endswith("_left") or v.stem.endswith("_right")
    ]

    print(f"ラベル付き動画数: {len(videos)}")

    if len(videos) == 0:
        print("ラベル付き動画がありません")
        return

    rows = []

    # --------------------------------------------------------
    # 解析
    # --------------------------------------------------------

    for i, video in enumerate(videos, 1):

        print()
        print("=" * 60)
        print(f"{i}/{len(videos)}")
        print(video.name)
        print("=" * 60)

        # --------------------------------------------
        # ラベル
        # --------------------------------------------

        if video.stem.endswith("_left"):
            hand = "left"

        elif video.stem.endswith("_right"):
            hand = "right"

        else:
            continue

        # --------------------------------------------
        # YOLO Pose
        # --------------------------------------------

        try:

            results = analyze_video(str(video))

            keypoints = extract_keypoints(results)

            print(f"取得フレーム数: {len(keypoints)}")

            if len(keypoints) < 20:

                print("⚠ フレーム不足")
                continue

            # ----------------------------------------
            # 特徴量
            # ----------------------------------------

            features = create_hand_features(keypoints)

            if features is None:
                continue

            # ----------------------------------------
            # メタ情報
            # ----------------------------------------

            features["video_name"] = video.name
            features["hand"] = hand

            rows.append(features)

            print(f"ラベル: {hand}")

        except Exception as e:

            print(f"⚠ エラー: {e}")

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    if len(rows) == 0:

        print("特徴量を作成できませんでした")
        return

    result = pd.DataFrame(rows)

    # video_nameを先頭
    columns = ["video_name", "hand"] + [
        c for c in result.columns if c not in ["video_name", "hand"]
    ]

    result = result[columns]

    result.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("利き腕分類用データセット作成完了")
    print("=" * 60)

    print(f"保存先: {OUTPUT_FILE}")
    print(f"データ数: {len(result)}")
    print(f"特徴量数: {len(result.columns) - 2}")

    print()
    print(result["hand"].value_counts())


if __name__ == "__main__":
    main()
