import numpy as np
import pandas as pd
from pathlib import Path

from pose import analyze_video

# ============================================================
# 設定
# ============================================================

VIDEO_DIR = Path("test_videos")


# ============================================================
# キーポイント抽出
# ============================================================


def extract_keypoints(results):

    rows = []

    for result in results:

        # ----------------------------------------------------
        # キーポイントが検出されなかった場合
        # ----------------------------------------------------
        if result.keypoints is None:
            rows.append([np.nan] * 12)
            continue

        xy = result.keypoints.xy

        if xy is None or len(xy) == 0:
            rows.append([np.nan] * 12)
            continue

        # 最初に検出された人物
        points = xy[0].cpu().numpy()

        # COCO Pose は17点
        if len(points) < 11:
            rows.append([np.nan] * 12)
            continue

        # ----------------------------------------------------
        # COCO Pose
        #
        # 5  左肩
        # 6  右肩
        # 7  左肘
        # 8  右肘
        # 9  左手首
        # 10 右手首
        # ----------------------------------------------------

        rows.append(
            [
                # 左肩
                points[5][0],
                points[5][1],
                # 右肩
                points[6][0],
                points[6][1],
                # 左肘
                points[7][0],
                points[7][1],
                # 右肘
                points[8][0],
                points[8][1],
                # 左手首
                points[9][0],
                points[9][1],
                # 右手首
                points[10][0],
                points[10][1],
            ]
        )

    return np.array(rows, dtype=float)


# ============================================================
# 欠損値の補間
# ============================================================


def interpolate(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    series = pd.Series(values)

    series = series.replace([np.inf, -np.inf], np.nan).interpolate(
        limit_direction="both"
    )

    return series.to_numpy()


# ============================================================
# 移動平均
# ============================================================


def smooth(values, window=5):

    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return values

    series = pd.Series(values)

    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


# ============================================================
# 速度計算
# ============================================================


def calculate_speed(x, y):

    x = interpolate(x)
    y = interpolate(y)

    if len(x) < 2:
        return np.array([])

    dx = np.diff(x)
    dy = np.diff(y)

    speed = np.sqrt(dx**2 + dy**2)

    return speed


# ============================================================
# 上位パーセンタイル
# ============================================================


def percentile(values, p=90):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, p))


# ============================================================
# 上位フレームの平均
# ============================================================


def top_mean(values, n=10):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    values = np.sort(values)

    n = min(n, len(values))

    return float(np.mean(values[-n:]))


# ============================================================
# 連続フレーム評価
# ============================================================


def consecutive_score(values, percentile_value):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0

    threshold = percentile_value

    max_count = 0
    current_count = 0

    for value in values:

        if value >= threshold:

            current_count += 1

            if current_count > max_count:
                max_count = current_count

        else:

            current_count = 0

    return max_count


# ============================================================
# 1動画を診断
# ============================================================


def diagnose(video_path, true_hand):

    print()
    print("=" * 70)
    print(f"動画: {video_path.name}")
    print(f"正解: {true_hand}")
    print("=" * 70)

    # --------------------------------------------------------
    # YOLO Pose
    # --------------------------------------------------------

    results = analyze_video(video_path)

    points = extract_keypoints(results)

    if points.size == 0:

        print("キーポイントが取得できませんでした。")

        return

    if points.shape[1] != 12:

        print(f"ERROR: キーポイント数がおかしいです: " f"{points.shape[1]}")

        return

    print(f"フレーム数: {len(points)}")

    # --------------------------------------------------------
    # 各座標
    # --------------------------------------------------------

    left_shoulder_x = interpolate(points[:, 0])
    left_shoulder_y = interpolate(points[:, 1])

    right_shoulder_x = interpolate(points[:, 2])
    right_shoulder_y = interpolate(points[:, 3])

    left_elbow_x = interpolate(points[:, 4])
    left_elbow_y = interpolate(points[:, 5])

    right_elbow_x = interpolate(points[:, 6])
    right_elbow_y = interpolate(points[:, 7])

    left_wrist_x = interpolate(points[:, 8])
    left_wrist_y = interpolate(points[:, 9])

    right_wrist_x = interpolate(points[:, 10])
    right_wrist_y = interpolate(points[:, 11])

    # --------------------------------------------------------
    # 肩の中心
    # --------------------------------------------------------

    shoulder_x = (left_shoulder_x + right_shoulder_x) / 2

    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # --------------------------------------------------------
    # 肩からの距離
    # --------------------------------------------------------

    left_wrist_dist = np.sqrt(
        (left_wrist_x - shoulder_x) ** 2 + (left_wrist_y - shoulder_y) ** 2
    )

    right_wrist_dist = np.sqrt(
        (right_wrist_x - shoulder_x) ** 2 + (right_wrist_y - shoulder_y) ** 2
    )

    left_elbow_dist = np.sqrt(
        (left_elbow_x - shoulder_x) ** 2 + (left_elbow_y - shoulder_y) ** 2
    )

    right_elbow_dist = np.sqrt(
        (right_elbow_x - shoulder_x) ** 2 + (right_elbow_y - shoulder_y) ** 2
    )

    # --------------------------------------------------------
    # 後半だけを見る
    #
    # 投球動作の後半に腕の動きが大きくなるため
    # --------------------------------------------------------

    start = int(len(points) * 0.50)

    left_wrist_dist = left_wrist_dist[start:]
    right_wrist_dist = right_wrist_dist[start:]

    left_elbow_dist = left_elbow_dist[start:]
    right_elbow_dist = right_elbow_dist[start:]

    # --------------------------------------------------------
    # 平滑化
    # --------------------------------------------------------

    left_wrist_dist = smooth(left_wrist_dist, window=5)

    right_wrist_dist = smooth(right_wrist_dist, window=5)

    left_elbow_dist = smooth(left_elbow_dist, window=5)

    right_elbow_dist = smooth(right_elbow_dist, window=5)

    # --------------------------------------------------------
    # 手首速度
    # --------------------------------------------------------

    left_wrist_speed = calculate_speed(left_wrist_x[start:], left_wrist_y[start:])

    right_wrist_speed = calculate_speed(right_wrist_x[start:], right_wrist_y[start:])

    # --------------------------------------------------------
    # 肘速度
    # --------------------------------------------------------

    left_elbow_speed = calculate_speed(left_elbow_x[start:], left_elbow_y[start:])

    right_elbow_speed = calculate_speed(right_elbow_x[start:], right_elbow_y[start:])

    # --------------------------------------------------------
    # 平滑化
    # --------------------------------------------------------

    left_wrist_speed = smooth(left_wrist_speed, window=5)

    right_wrist_speed = smooth(right_wrist_speed, window=5)

    left_elbow_speed = smooth(left_elbow_speed, window=5)

    right_elbow_speed = smooth(right_elbow_speed, window=5)

    # ========================================================
    # 距離の評価
    # ========================================================

    lw_dist = percentile(left_wrist_dist, 90)

    rw_dist = percentile(right_wrist_dist, 90)

    le_dist = percentile(left_elbow_dist, 90)

    re_dist = percentile(right_elbow_dist, 90)

    # ========================================================
    # 速度の評価
    # ========================================================

    lw_speed = percentile(left_wrist_speed, 90)

    rw_speed = percentile(right_wrist_speed, 90)

    le_speed = percentile(left_elbow_speed, 90)

    re_speed = percentile(right_elbow_speed, 90)

    # ========================================================
    # 連続フレーム
    # ========================================================

    lw_consecutive = consecutive_score(left_wrist_speed, lw_speed)

    rw_consecutive = consecutive_score(right_wrist_speed, rw_speed)

    le_consecutive = consecutive_score(left_elbow_speed, le_speed)

    re_consecutive = consecutive_score(right_elbow_speed, re_speed)

    # ========================================================
    # 表示
    # ========================================================

    print()
    print("【肩からの距離：90パーセンタイル】")

    print(f"左手首: {lw_dist:.3f}")

    print(f"右手首: {rw_dist:.3f}")

    print(f"左肘:   {le_dist:.3f}")

    print(f"右肘:   {re_dist:.3f}")

    print()
    print("【動きの速さ：90パーセンタイル】")

    print(f"左手首: {lw_speed:.3f}")

    print(f"右手首: {rw_speed:.3f}")

    print(f"左肘:   {le_speed:.3f}")

    print(f"右肘:   {re_speed:.3f}")

    print()
    print("【連続フレーム数】")

    print(f"左手首: {lw_consecutive}")

    print(f"右手首: {rw_consecutive}")

    print(f"左肘:   {le_consecutive}")

    print(f"右肘:   {re_consecutive}")

    # ========================================================
    # 総合スコア
    # ========================================================

    left_score = lw_speed + 0.5 * le_speed + 0.2 * lw_dist + 0.1 * le_dist

    right_score = rw_speed + 0.5 * re_speed + 0.2 * rw_dist + 0.1 * re_dist

    print()
    print("【総合スコア】")

    print(f"左:  {left_score:.3f}")

    print(f"右:  {right_score:.3f}")

    # ========================================================
    # 判定
    # ========================================================

    if left_score > right_score:

        ratio = left_score / max(right_score, 1e-6)

        if ratio >= 1.10:
            predicted = "left"
        else:
            predicted = "unknown"

    else:

        ratio = right_score / max(left_score, 1e-6)

        if ratio >= 1.10:
            predicted = "right"
        else:
            predicted = "unknown"

    print()
    print(f"推定: {predicted}")

    print(f"スコア比: {ratio:.3f}")

    # ========================================================
    # 正解判定
    # ========================================================

    if predicted == true_hand:

        print("結果: ○ 正解")

    elif predicted == "unknown":

        print("結果: △ unknown")

    else:

        print("結果: × 不正解")


# ============================================================
# メイン
# ============================================================


def main():

    print()
    print(f"動画フォルダ: " f"{VIDEO_DIR.resolve()}")

    if not VIDEO_DIR.exists():

        print("動画フォルダが存在しません。")

        return

    # --------------------------------------------------------
    # 今回テストする動画
    #
    # ファイル名に含まれる速度を利用して検索する
    # --------------------------------------------------------

    test_patterns = [
        ("mlb_001_81.1mph_", "right"),
        ("mlb_001_81.5mph_", "right"),
        ("mlb_001_82.5mph_", "right"),
        ("mlb_002_81.5mph_", "right"),
        ("mlb_002_81.8mph_", "right"),
        ("mlb_002_81.9mph_", "right"),
        ("mlb_012_83.0mph_", "left"),
        ("mlb_012_86.6mph_", "left"),
        ("mlb_013_86.6mph_", "left"),
        ("mlb_013_90.8mph_", "left"),
        ("mlb_014_86.8mph_", "left"),
        ("mlb_014_88.2mph_", "left"),
    ]

    videos = []

    # --------------------------------------------------------
    # 動画を検索
    # --------------------------------------------------------

    for prefix, true_hand in test_patterns:

        matches = sorted(VIDEO_DIR.glob(prefix + "*.mp4"))

        if len(matches) == 0:

            print()
            print(f"動画が見つかりません: " f"{prefix}*.mp4")

            continue

        # 同じ条件の動画が複数ある場合は最初の1本
        videos.append((matches[0], true_hand))

    # --------------------------------------------------------
    # 見つかった動画を表示
    # --------------------------------------------------------

    print()
    print(f"診断対象動画数: {len(videos)}")

    if len(videos) == 0:

        print("診断できる動画がありません。")

        return

    # --------------------------------------------------------
    # 全動画を診断
    # --------------------------------------------------------

    correct = 0
    total = 0

    for video_path, true_hand in videos:

        diagnose(video_path, true_hand)

        total += 1

        # 個別の判定をここでは再取得しないため、
        # 正解率は最後に手動確認用として表示
        #
        # 必要なら後で自動集計版に変更可能

    print()
    print("=" * 70)
    print("診断終了")
    print("=" * 70)

    print(f"診断した動画数: {total}")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()
