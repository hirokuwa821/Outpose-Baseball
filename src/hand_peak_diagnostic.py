import sys
from pathlib import Path

import numpy as np

# プロジェクトルート
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pose import analyze_video

VIDEO_DIR = ROOT / "test_videos"


# ============================================================
# テスト動画
# ============================================================

VIDEOS = [
    # 右投げ
    ("mlb_001_81.1mph_CGI1SSOSP466.mp4", "right"),
    ("mlb_001_81.5mph_KPY8PN4UY84W.mp4", "right"),
    ("mlb_001_82.5mph_HQCPRB2AYCFZ.mp4", "right"),
    ("mlb_002_81.5mph_KPY8PN4UY84W.mp4", "right"),
    ("mlb_002_81.8mph_W64AW7R2SWGS.mp4", "right"),
    ("mlb_002_81.9mph_MM9QJ9EPUPHF.mp4", "right"),
    # 左投げ
    ("mlb_012_83.0mph_HJWRVIP9C2D8.mp4", "left"),
    ("mlb_012_86.6mph_YJ9DKED18D5W.mp4", "left"),
    ("mlb_013_86.6mph_RD0X9UHNRR22.mp4", "left"),
    ("mlb_013_90.8mph_LL3M40DBY7AK.mp4", "left"),
    ("mlb_014_86.8mph_475IVXI4IRDZ.mp4", "left"),
    ("mlb_014_88.2mph_2BBYMCKBDVB5.mp4", "left"),
]


# ============================================================
# Keypoint
#
# 5  左肩
# 6  右肩
# 7  左肘
# 8  右肘
# 9  左手首
# 10 右手首
# ============================================================


def get_point(result, index):

    if result.keypoints is None:
        return None

    if result.keypoints.xy is None:
        return None

    if len(result.keypoints.xy) == 0:
        return None

    points = result.keypoints.xy[0]

    if len(points) <= index:
        return None

    x = float(points[index][0])
    y = float(points[index][1])

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return np.array([x, y], dtype=float)


def convert_points(points):

    output = []

    for p in points:

        if p is None:
            output.append([np.nan, np.nan])
        else:
            output.append(p)

    return np.array(output, dtype=float)


def calc_speed(points):

    if len(points) < 2:
        return np.array([])

    diff = np.diff(points, axis=0)

    speed = np.linalg.norm(diff, axis=1)

    return speed


def get_peak_info(speed, start_frame):

    speed = np.asarray(speed)

    valid = np.isfinite(speed)

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]

    local_index = valid_indices[np.argmax(speed[valid])]

    peak_speed = float(speed[local_index])

    # np.diffなので元動画では+1フレーム
    frame = start_frame + local_index + 1

    return frame, peak_speed


def percentile(values, p):

    values = np.asarray(values)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, p))


# ============================================================
# 1動画を解析
# ============================================================


def analyze_one(filename, label):

    path = VIDEO_DIR / filename

    print()
    print("=" * 75)
    print(filename)
    print("正解:", label)
    print("=" * 75)

    if not path.exists():

        print("動画がありません:")
        print(path)

        return

    results = analyze_video(path)

    # --------------------------------------------------------
    # keypoint保存
    # --------------------------------------------------------

    left_shoulder = []
    right_shoulder = []

    left_elbow = []
    right_elbow = []

    left_wrist = []
    right_wrist = []

    for result in results:

        left_shoulder.append(get_point(result, 5))
        right_shoulder.append(get_point(result, 6))

        left_elbow.append(get_point(result, 7))
        right_elbow.append(get_point(result, 8))

        left_wrist.append(get_point(result, 9))
        right_wrist.append(get_point(result, 10))

    ls = convert_points(left_shoulder)
    rs = convert_points(right_shoulder)

    le = convert_points(left_elbow)
    re = convert_points(right_elbow)

    lw = convert_points(left_wrist)
    rw = convert_points(right_wrist)

    n = len(lw)

    print()
    print("総フレーム数:", n)

    if n < 20:

        print("フレーム数不足")

        return

    # --------------------------------------------------------
    # 後半50%
    # --------------------------------------------------------

    start = int(n * 0.50)

    print("解析開始フレーム:", start)
    print("解析範囲:", f"{start} ～ {n - 1}")

    lw_tail = lw[start:]
    rw_tail = rw[start:]

    le_tail = le[start:]
    re_tail = re[start:]

    # --------------------------------------------------------
    # 速度
    # --------------------------------------------------------

    left_wrist_speed = calc_speed(lw_tail)
    right_wrist_speed = calc_speed(rw_tail)

    left_elbow_speed = calc_speed(le_tail)
    right_elbow_speed = calc_speed(re_tail)

    # --------------------------------------------------------
    # 速度ピーク
    # --------------------------------------------------------

    left_wrist_peak = get_peak_info(left_wrist_speed, start)

    right_wrist_peak = get_peak_info(right_wrist_speed, start)

    left_elbow_peak = get_peak_info(left_elbow_speed, start)

    right_elbow_peak = get_peak_info(right_elbow_speed, start)

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print()
    print("【手首速度ピーク】")

    if left_wrist_peak:

        frame, speed = left_wrist_peak

        print(
            f"左手首 : frame {frame:4d} / "
            f"{frame / n * 100:6.2f}% / "
            f"速度 {speed:8.3f}"
        )

    else:

        print("左手首 : データなし")

    if right_wrist_peak:

        frame, speed = right_wrist_peak

        print(
            f"右手首 : frame {frame:4d} / "
            f"{frame / n * 100:6.2f}% / "
            f"速度 {speed:8.3f}"
        )

    else:

        print("右手首 : データなし")

    print()
    print("【肘速度ピーク】")

    if left_elbow_peak:

        frame, speed = left_elbow_peak

        print(
            f"左肘   : frame {frame:4d} / "
            f"{frame / n * 100:6.2f}% / "
            f"速度 {speed:8.3f}"
        )

    else:

        print("左肘   : データなし")

    if right_elbow_peak:

        frame, speed = right_elbow_peak

        print(
            f"右肘   : frame {frame:4d} / "
            f"{frame / n * 100:6.2f}% / "
            f"速度 {speed:8.3f}"
        )

    else:

        print("右肘   : データなし")

    # --------------------------------------------------------
    # 上位10%速度
    # --------------------------------------------------------

    lw90 = percentile(left_wrist_speed, 90)
    rw90 = percentile(right_wrist_speed, 90)

    le90 = percentile(left_elbow_speed, 90)
    re90 = percentile(right_elbow_speed, 90)

    print()
    print("【速度 上位10%】")

    print(f"左手首 : {lw90:.3f}")
    print(f"右手首 : {rw90:.3f}")

    print(f"左肘   : {le90:.3f}")
    print(f"右肘   : {re90:.3f}")

    # --------------------------------------------------------
    # ピーク時刻の左右差
    # --------------------------------------------------------

    print()
    print("【ピークタイミング差】")

    if left_wrist_peak and right_wrist_peak:

        lf = left_wrist_peak[0]
        rf = right_wrist_peak[0]

        print(f"手首ピーク差 : {abs(lf - rf)} frame")

        print(f"左 - 右      : {lf - rf:+d} frame")

    if left_elbow_peak and right_elbow_peak:

        lf = left_elbow_peak[0]
        rf = right_elbow_peak[0]

        print(f"肘ピーク差   : {abs(lf - rf)} frame")

        print(f"左 - 右      : {lf - rf:+d} frame")

    # --------------------------------------------------------
    # 手首＋肘の総合速度
    # --------------------------------------------------------

    combined_left = left_wrist_speed + 0.5 * left_elbow_speed

    combined_right = right_wrist_speed + 0.5 * right_elbow_speed

    left_combined_peak = get_peak_info(combined_left, start)

    right_combined_peak = get_peak_info(combined_right, start)

    print()
    print("【手首＋肘 総合速度ピーク】")

    if left_combined_peak:

        frame, speed = left_combined_peak

        print(f"左 : frame {frame:4d} / " f"{frame / n * 100:6.2f}% / " f"{speed:8.3f}")

    if right_combined_peak:

        frame, speed = right_combined_peak

        print(f"右 : frame {frame:4d} / " f"{frame / n * 100:6.2f}% / " f"{speed:8.3f}")


# ============================================================
# メイン
# ============================================================


def main():

    print("=" * 75)
    print("左右判定：ピークタイミング診断")
    print("=" * 75)

    for filename, label in VIDEOS:

        analyze_one(filename, label)

    print()
    print("=" * 75)
    print("診断終了")
    print("=" * 75)


if __name__ == "__main__":

    main()
