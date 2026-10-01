import sys
from pathlib import Path

import numpy as np

# src フォルダから実行しても、プロジェクトルートから実行しても動くようにする
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pose import analyze_video

# ============================================================
# 調べる動画
# ============================================================

VIDEO_DIR = ROOT / "test_videos"

VIDEOS = [
    # 右投げ
    "mlb_001_81.1mph_CGI1SSOSP466.mp4",
    "mlb_001_81.5mph_KPY8PN4UY84W.mp4",
    "mlb_001_82.5mph_HQCPRB2AYCFZ.mp4",
    "mlb_002_81.5mph_KPY8PN4UY84W.mp4",
    "mlb_002_81.8mph_W64AW7R2SWGS.mp4",
    "mlb_002_81.9mph_MM9QJ9EPUPHF.mp4",
    # 左投げ
    "mlb_012_83.0mph_HJWRVIP9C2D8.mp4",
    "mlb_012_86.6mph_YJ9DKED18D5W.mp4",
    "mlb_013_86.6mph_RD0X9UHNRR22.mp4",
    "mlb_013_90.8mph_LL3M40DBY7AK.mp4",
    "mlb_014_86.8mph_475IVXI4IRDZ.mp4",
    "mlb_014_88.2mph_2BBYMCKBDVB5.mp4",
]


# ============================================================
# YOLO keypoint番号
#
# 5  = 左肩
# 6  = 右肩
# 7  = 左肘
# 8  = 右肘
# 9  = 左手首
# 10 = 右手首
# ============================================================


def get_xy(result, index):
    """YOLO結果から指定したkeypointのx,yを取得"""

    if result.keypoints is None:
        return None

    if result.keypoints.xy is None:
        return None

    xy = result.keypoints.xy

    if len(xy) == 0:
        return None

    # 最初の人物
    points = xy[0]

    if len(points) <= index:
        return None

    x = float(points[index][0])
    y = float(points[index][1])

    if not np.isfinite(x) or not np.isfinite(y):
        return None

    return np.array([x, y], dtype=float)


def distance(a, b):
    if a is None or b is None:
        return np.nan

    return float(np.linalg.norm(a - b))


def velocity(points):
    """座標列からフレーム間移動量を計算"""

    if len(points) < 2:
        return np.array([])

    points = np.asarray(points)

    diff = np.diff(points, axis=0)

    return np.linalg.norm(diff, axis=1)


def percentile(values, p):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, p))


def analyze_one(video_path, label):

    print()
    print("=" * 70)
    print(video_path.name)
    print("正解:", label)
    print("=" * 70)

    results = analyze_video(video_path)

    left_shoulder = []
    right_shoulder = []
    left_elbow = []
    right_elbow = []
    left_wrist = []
    right_wrist = []

    for result in results:

        ls = get_xy(result, 5)
        rs = get_xy(result, 6)

        le = get_xy(result, 7)
        re = get_xy(result, 8)

        lw = get_xy(result, 9)
        rw = get_xy(result, 10)

        left_shoulder.append(ls)
        right_shoulder.append(rs)

        left_elbow.append(le)
        right_elbow.append(re)

        left_wrist.append(lw)
        right_wrist.append(rw)

    # --------------------------------------------------------
    # numpy配列化
    # --------------------------------------------------------

    def clean(points):
        valid = []

        for p in points:
            if p is None:
                valid.append([np.nan, np.nan])
            else:
                valid.append(p)

        return np.array(valid, dtype=float)

    ls = clean(left_shoulder)
    rs = clean(right_shoulder)
    le = clean(left_elbow)
    re = clean(right_elbow)
    lw = clean(left_wrist)
    rw = clean(right_wrist)

    n = len(ls)

    if n < 20:
        print("フレーム数不足")
        return

    # --------------------------------------------------------
    # 肩中心
    # --------------------------------------------------------

    shoulder_center = (ls + rs) / 2

    # --------------------------------------------------------
    # 肩幅
    # --------------------------------------------------------

    shoulder_width = np.linalg.norm(rs - ls, axis=1)

    # --------------------------------------------------------
    # 肩中心から手首までの距離
    # --------------------------------------------------------

    left_wrist_dist = np.linalg.norm(lw - shoulder_center, axis=1)

    right_wrist_dist = np.linalg.norm(rw - shoulder_center, axis=1)

    # --------------------------------------------------------
    # 肩中心から肘までの距離
    # --------------------------------------------------------

    left_elbow_dist = np.linalg.norm(le - shoulder_center, axis=1)

    right_elbow_dist = np.linalg.norm(re - shoulder_center, axis=1)

    # --------------------------------------------------------
    # 肩幅で正規化
    # カメラとの距離の影響を減らす
    # --------------------------------------------------------

    left_wrist_norm = left_wrist_dist / shoulder_width
    right_wrist_norm = right_wrist_dist / shoulder_width

    left_elbow_norm = left_elbow_dist / shoulder_width
    right_elbow_norm = right_elbow_dist / shoulder_width

    # --------------------------------------------------------
    # 後半50%
    # --------------------------------------------------------

    start = int(n * 0.50)

    lw_tail = lw[start:]
    rw_tail = rw[start:]

    left_wrist_norm_tail = left_wrist_norm[start:]
    right_wrist_norm_tail = right_wrist_norm[start:]

    left_elbow_norm_tail = left_elbow_norm[start:]
    right_elbow_norm_tail = right_elbow_norm[start:]

    # --------------------------------------------------------
    # 移動速度
    # --------------------------------------------------------

    left_speed = velocity(lw_tail)
    right_speed = velocity(rw_tail)

    left_elbow_speed = velocity(le[start:])
    right_elbow_speed = velocity(re[start:])

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    print()

    print("【手首】")
    print(f"左 手首距離 最大     : " f"{percentile(left_wrist_norm_tail, 100):.3f}")
    print(f"右 手首距離 最大     : " f"{percentile(right_wrist_norm_tail, 100):.3f}")

    print(f"左 手首距離 上位10%平均: " f"{percentile(left_wrist_norm_tail, 90):.3f}")
    print(f"右 手首距離 上位10%平均: " f"{percentile(right_wrist_norm_tail, 90):.3f}")

    print(f"左 手首速度 上位10%   : " f"{percentile(left_speed, 90):.3f}")
    print(f"右 手首速度 上位10%   : " f"{percentile(right_speed, 90):.3f}")

    print()

    print("【肘】")

    print(f"左 肘距離 上位10%     : " f"{percentile(left_elbow_norm_tail, 90):.3f}")
    print(f"右 肘距離 上位10%     : " f"{percentile(right_elbow_norm_tail, 90):.3f}")

    print(f"左 肘速度 上位10%     : " f"{percentile(left_elbow_speed, 90):.3f}")
    print(f"右 肘速度 上位10%     : " f"{percentile(right_elbow_speed, 90):.3f}")

    # --------------------------------------------------------
    # 左右比
    # --------------------------------------------------------

    left_w = percentile(left_wrist_norm_tail, 90)
    right_w = percentile(right_wrist_norm_tail, 90)

    left_ws = percentile(left_speed, 90)
    right_ws = percentile(right_speed, 90)

    left_e = percentile(left_elbow_norm_tail, 90)
    right_e = percentile(right_elbow_norm_tail, 90)

    left_es = percentile(left_elbow_speed, 90)
    right_es = percentile(right_elbow_speed, 90)

    print()
    print("【左右比】")

    if np.isfinite(left_w) and np.isfinite(right_w):
        print(f"手首距離 左/右 : {left_w / right_w:.3f}")
        print(f"手首距離 右/左 : {right_w / left_w:.3f}")

    if np.isfinite(left_ws) and np.isfinite(right_ws):
        print(f"手首速度 左/右 : {left_ws / right_ws:.3f}")
        print(f"手首速度 右/左 : {right_ws / left_ws:.3f}")

    if np.isfinite(left_e) and np.isfinite(right_e):
        print(f"肘距離   左/右 : {left_e / right_e:.3f}")
        print(f"肘距離   右/左 : {right_e / left_e:.3f}")

    if np.isfinite(left_es) and np.isfinite(right_es):
        print(f"肘速度   左/右 : {left_es / right_es:.3f}")
        print(f"肘速度   右/左 : {right_es / left_es:.3f}")


def main():

    print("=" * 70)
    print("左右判定 診断")
    print("=" * 70)

    for filename in VIDEOS:

        path = VIDEO_DIR / filename

        if not path.exists():
            print()
            print("動画がありません:", path)
            continue

        # 正解ラベル
        if (
            filename.startswith("mlb_012")
            or filename.startswith("mlb_013")
            or filename.startswith("mlb_014")
        ):
            label = "left"
        else:
            label = "right"

        analyze_one(path, label)

    print()
    print("=" * 70)
    print("診断終了")
    print("=" * 70)


if __name__ == "__main__":
    main()
