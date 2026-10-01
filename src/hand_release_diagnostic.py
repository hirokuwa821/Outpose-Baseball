import numpy as np
import pandas as pd
from pathlib import Path

from pose import analyze_video

VIDEO_DIR = Path("test_videos")


# ============================================================
# キーポイント抽出
# ============================================================


def extract_keypoints(results):

    rows = []

    for result in results:

        if result.keypoints is None:
            rows.append([np.nan] * 12)
            continue

        xy = result.keypoints.xy

        if xy is None or len(xy) == 0:
            rows.append([np.nan] * 12)
            continue

        points = xy[0].cpu().numpy()

        if len(points) < 11:
            rows.append([np.nan] * 12)
            continue

        rows.append(
            [
                points[5][0],  # 左肩 x
                points[5][1],  # 左肩 y
                points[6][0],  # 右肩 x
                points[6][1],  # 右肩 y
                points[7][0],  # 左肘 x
                points[7][1],  # 左肘 y
                points[8][0],  # 右肘 x
                points[8][1],  # 右肘 y
                points[9][0],  # 左手首 x
                points[9][1],  # 左手首 y
                points[10][0],  # 右手首 x
                points[10][1],  # 右手首 y
            ]
        )

    return np.array(rows, dtype=float)


# ============================================================
# 欠損値補間
# ============================================================


def interpolate(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    s = pd.Series(values)

    s = s.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")

    return s.to_numpy()


# ============================================================
# 平滑化
# ============================================================


def smooth(values, window=5):

    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return values

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


# ============================================================
# 速度
# ============================================================


def velocity(x, y):

    x = interpolate(x)
    y = interpolate(y)

    if len(x) < 2:
        return np.array([])

    dx = np.diff(x)
    dy = np.diff(y)

    return np.sqrt(dx**2 + dy**2)


# ============================================================
# 角度
# ============================================================


def angle(a_x, a_y, b_x, b_y):

    dx = b_x - a_x
    dy = b_y - a_y

    return np.degrees(np.arctan2(dy, dx))


# ============================================================
# 角速度
# ============================================================


def angle_velocity(angles):

    angles = np.asarray(angles)

    if len(angles) < 2:
        return np.array([])

    diff = np.diff(angles)

    # -180° / +180° をまたぐ問題を補正
    diff = (diff + 180) % 360 - 180

    return np.abs(diff)


# ============================================================
# リリース候補を探す
# ============================================================


def detect_release_frame(wrist_x, wrist_y, elbow_x, elbow_y):

    wrist_speed = velocity(wrist_x, wrist_y)

    elbow_speed = velocity(elbow_x, elbow_y)

    if len(wrist_speed) == 0:
        return 0

    # 手首＋肘の動き
    combined = wrist_speed + 0.5 * elbow_speed

    # 後半60%だけから探す
    start = int(len(combined) * 0.40)

    search = combined[start:]

    if len(search) == 0:
        return int(len(combined) // 2)

    local_index = int(np.argmax(search))

    return start + local_index


# ============================================================
# リリース周辺の評価
# ============================================================


def release_metrics(points, release_frame):

    # ----------------------------------------
    # 各座標
    # ----------------------------------------

    lsx = interpolate(points[:, 0])
    lsy = interpolate(points[:, 1])

    rsx = interpolate(points[:, 2])
    rsy = interpolate(points[:, 3])

    lex = interpolate(points[:, 4])
    ley = interpolate(points[:, 5])

    rex = interpolate(points[:, 6])
    rey = interpolate(points[:, 7])

    lwx = interpolate(points[:, 8])
    lwy = interpolate(points[:, 9])

    rwx = interpolate(points[:, 10])
    rwy = interpolate(points[:, 11])

    # ----------------------------------------
    # 肩中心
    # ----------------------------------------

    sx = (lsx + rsx) / 2
    sy = (lsy + rsy) / 2

    # ----------------------------------------
    # 肘角度
    # 肩 → 肘 → 手首
    # ----------------------------------------

    left_angle = angle(lex, ley, lwx, lwy)

    right_angle = angle(rex, rey, rwx, rwy)

    left_angle = smooth(left_angle)

    right_angle = smooth(right_angle)

    left_angle_speed = angle_velocity(left_angle)

    right_angle_speed = angle_velocity(right_angle)

    # ----------------------------------------
    # 手首速度
    # ----------------------------------------

    left_wrist_speed = velocity(lwx, lwy)

    right_wrist_speed = velocity(rwx, rwy)

    left_wrist_speed = smooth(left_wrist_speed)

    right_wrist_speed = smooth(right_wrist_speed)

    # ----------------------------------------
    # 肘速度
    # ----------------------------------------

    left_elbow_speed = velocity(lex, ley)

    right_elbow_speed = velocity(rex, rey)

    left_elbow_speed = smooth(left_elbow_speed)

    right_elbow_speed = smooth(right_elbow_speed)

    # ----------------------------------------
    # リリース前後の範囲
    # ----------------------------------------

    start = max(0, release_frame - 8)

    end = min(len(points) - 1, release_frame + 8)

    # ----------------------------------------
    # 速度
    # ----------------------------------------

    lw = left_wrist_speed[max(0, start - 1) : end]

    rw = right_wrist_speed[max(0, start - 1) : end]

    le = left_elbow_speed[max(0, start - 1) : end]

    re = right_elbow_speed[max(0, start - 1) : end]

    la = left_angle_speed[max(0, start - 1) : end]

    ra = right_angle_speed[max(0, start - 1) : end]

    # ----------------------------------------
    # それぞれの最大値
    # ----------------------------------------

    metrics = {
        "left_wrist_max": np.max(lw) if len(lw) else 0,
        "right_wrist_max": np.max(rw) if len(rw) else 0,
        "left_wrist_mean": np.mean(lw) if len(lw) else 0,
        "right_wrist_mean": np.mean(rw) if len(rw) else 0,
        "left_elbow_max": np.max(le) if len(le) else 0,
        "right_elbow_max": np.max(re) if len(re) else 0,
        "left_elbow_mean": np.mean(le) if len(le) else 0,
        "right_elbow_mean": np.mean(re) if len(re) else 0,
        "left_angle_max": np.max(la) if len(la) else 0,
        "right_angle_max": np.max(ra) if len(ra) else 0,
        "left_angle_mean": np.mean(la) if len(la) else 0,
        "right_angle_mean": np.mean(ra) if len(ra) else 0,
    }

    return metrics


# ============================================================
# 1動画を診断
# ============================================================


def diagnose(video_path, true_hand):

    print()
    print("=" * 70)
    print(f"動画: {video_path.name}")
    print(f"正解: {true_hand}")
    print("=" * 70)

    # YOLO Pose
    results = analyze_video(video_path)

    points = extract_keypoints(results)

    if points.size == 0:

        print("キーポイントが取得できませんでした。")

        return None

    print(f"フレーム数: {len(points)}")

    # ========================================================
    # 左右の手首からリリース候補を探す
    # ========================================================

    left_release = detect_release_frame(
        points[:, 8], points[:, 9], points[:, 4], points[:, 5]
    )

    right_release = detect_release_frame(
        points[:, 10], points[:, 11], points[:, 6], points[:, 7]
    )

    # 左右で候補が離れすぎないようにする
    release_frame = int((left_release + right_release) / 2)

    print()
    print(f"左腕リリース候補: " f"{left_release}")

    print(f"右腕リリース候補: " f"{right_release}")

    print(f"平均リリース候補: " f"{release_frame}")

    print(f"動画に対する位置: " f"{release_frame / len(points) * 100:.1f}%")

    # ========================================================
    # リリース周辺を分析
    # ========================================================

    metrics = release_metrics(points, release_frame)

    print()
    print("【リリース前後の手首速度】")

    print(f"左 最大: " f"{metrics['left_wrist_max']:.3f}")

    print(f"右 最大: " f"{metrics['right_wrist_max']:.3f}")

    print(f"左 平均: " f"{metrics['left_wrist_mean']:.3f}")

    print(f"右 平均: " f"{metrics['right_wrist_mean']:.3f}")

    print()
    print("【リリース前後の肘速度】")

    print(f"左 最大: " f"{metrics['left_elbow_max']:.3f}")

    print(f"右 最大: " f"{metrics['right_elbow_max']:.3f}")

    print(f"左 平均: " f"{metrics['left_elbow_mean']:.3f}")

    print(f"右 平均: " f"{metrics['right_elbow_mean']:.3f}")

    print()
    print("【リリース前後の腕角度変化】")

    print(f"左 最大: " f"{metrics['left_angle_max']:.3f}")

    print(f"右 最大: " f"{metrics['right_angle_max']:.3f}")

    print(f"左 平均: " f"{metrics['left_angle_mean']:.3f}")

    print(f"右 平均: " f"{metrics['right_angle_mean']:.3f}")

    # ========================================================
    # 仮スコア
    # ========================================================

    left_score = (
        metrics["left_wrist_max"]
        + 0.5 * metrics["left_elbow_max"]
        + 0.3 * metrics["left_angle_max"]
    )

    right_score = (
        metrics["right_wrist_max"]
        + 0.5 * metrics["right_elbow_max"]
        + 0.3 * metrics["right_angle_max"]
    )

    print()
    print("【リリース動作スコア】")

    print(f"左: {left_score:.3f}")

    print(f"右: {right_score:.3f}")

    if left_score > right_score:

        ratio = left_score / max(right_score, 1e-6)

        predicted = "left" if ratio >= 1.10 else "unknown"

    else:

        ratio = right_score / max(left_score, 1e-6)

        predicted = "right" if ratio >= 1.10 else "unknown"

    print()
    print(f"仮判定: {predicted}")

    print(f"スコア比: {ratio:.3f}")

    if predicted == true_hand:

        print("結果: ○ 正解")

    elif predicted == "unknown":

        print("結果: △ unknown")

    else:

        print("結果: × 不正解")

    return {
        "video": video_path.name,
        "true_hand": true_hand,
        "predicted": predicted,
        "ratio": ratio,
        "left_score": left_score,
        "right_score": right_score,
        "release_frame": release_frame,
    }


# ============================================================
# メイン
# ============================================================


def main():

    print()
    print(f"動画フォルダ: " f"{VIDEO_DIR.resolve()}")

    if not VIDEO_DIR.exists():

        print("動画フォルダが存在しません。")

        return

    # ========================================================
    # テスト動画
    # ========================================================

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

    for prefix, true_hand in test_patterns:

        matches = sorted(VIDEO_DIR.glob(prefix + "*.mp4"))

        if len(matches) == 0:

            print(f"動画が見つかりません: " f"{prefix}*.mp4")

            continue

        videos.append((matches[0], true_hand))

    print()
    print(f"診断対象動画数: " f"{len(videos)}")

    # ========================================================
    # 全動画
    # ========================================================

    results = []

    for video_path, true_hand in videos:

        result = diagnose(video_path, true_hand)

        if result is not None:
            results.append(result)

    # ========================================================
    # 最終結果
    # ========================================================

    print()
    print("=" * 70)
    print("最終結果")
    print("=" * 70)

    correct = 0
    wrong = 0
    unknown = 0

    for result in results:

        if result["predicted"] == result["true_hand"]:

            correct += 1

        elif result["predicted"] == "unknown":

            unknown += 1

        else:

            wrong += 1

        print(
            f"{result['video']}: "
            f"正解={result['true_hand']} "
            f"推定={result['predicted']} "
            f"比={result['ratio']:.3f}"
        )

    total = len(results)

    print()
    print(f"正解: {correct}")

    print(f"不正解: {wrong}")

    print(f"unknown: {unknown}")

    print(f"合計: {total}")

    if total > 0:

        print(f"正解率: " f"{correct / total * 100:.1f}%")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()
