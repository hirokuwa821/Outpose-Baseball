from pathlib import Path
import pandas as pd
import re

from pose import analyze_video
from export_csv import export_keypoints
from release import detect_release
from features import create_features
from dataset import save_dataset

# ==============================
# パス設定
# ==============================


BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "test_videos"
def analyze_one_video(video_path, speed_mph):

    video = Path(video_path)

    speed_kmh = speed_mph * 1.609344

    print()
    print("=" * 70)
    print(f"解析中: {video.name}")
    print(f"球速: {speed_mph:.1f} mph")
    print(f"球速: {speed_kmh:.2f} km/h")
    print("=" * 70)

    # ① Pose解析
    results = analyze_video(str(video))

    # ② キーポイントCSV保存
    export_keypoints(results)

    keypoints_path = BASE_DIR / "output" / "keypoints.csv"

    if keypoints_path.exists():

        keypoints = pd.read_csv(keypoints_path)

        print(f"keypointsフレーム数: {len(keypoints)}")

    # ③ リリースフレーム検出
    release = detect_release("left")

    print(f"リリースフレーム: {release}")

    if release is None:

        print("リリースフレームが検出できなかったためスキップします")

        return False

    # ④ 特徴量作成
    release = create_features("left")

    if release is None:

        print("特徴量作成に失敗しました")

        return False

    # ⑤ dataset.csvへ保存
    save_dataset(speed_kmh, release, video.name)

    print(f"完了: {video.name}")

    return True


def get_speed_from_filename(filename):

    # 例:
    # mlb_030_97.5mph_GUGJK0KSVD7F.mp4

    match = re.search(r"_(\d+(?:\.\d+)?)mph_", filename)

    if match:
        return float(match.group(1))

    return None


def main():

    # MLB動画だけ取得
    videos = sorted(VIDEO_DIR.glob("mlb_*.mp4"))

    print("=" * 70)
    print(f"MLB投球動画: {len(videos)}本")
    print("=" * 70)

    if len(videos) == 0:
        print("mlb_*.mp4 が見つかりません")
        return

    # ==========================================
    # すでに解析済みの動画を確認
    # ==========================================

    dataset_path = BASE_DIR / "output" / "dataset.csv"

    analyzed_videos = set()

    if dataset_path.exists():

        data = pd.read_csv(dataset_path)

        if "video_name" in data.columns:
            analyzed_videos = set(data["video_name"].dropna().astype(str))

    print(f"すでに解析済み: {len(analyzed_videos)}本")
    print(f"今回解析する候補: {len(videos) - len(analyzed_videos)}本")

    # ==========================================
    # 解析
    # ==========================================

    success = 0
    failed = 0
    skipped = 0

    for i, video in enumerate(videos, 1):

        print()
        print(f"===== {i}/{len(videos)} =====")

        # ==========================================
        # すでに解析済みならスキップ
        # ==========================================

        if video.name in analyzed_videos:

            print(f"解析済み → スキップ: {video.name}")
            skipped += 1
            continue

        # ==========================================
        # 球速取得
        # ==========================================

        speed_mph = get_speed_from_filename(video.name)

        if speed_mph is None:

            print(f"球速をファイル名から取得できません: {video.name}")

            failed += 1
            continue

        # ==========================================
        # 動画解析
        # ==========================================

        try:

            result = analyze_one_video(video, speed_mph)

            if result:

                success += 1

                # 今回解析した動画を記録
                analyzed_videos.add(video.name)

            else:

                failed += 1

        except Exception as e:

            print()
            print("!!! この動画でエラーが発生しました !!!")
            print(video.name)
            print(f"エラー: {e}")

            failed += 1

    # ==========================================
    # 結果
    # ==========================================

    print()
    print("=" * 70)
    print("全動画の解析終了")
    print("=" * 70)

    print(f"今回の成功: {success}")
    print(f"スキップ: {skipped}")
    print(f"失敗: {failed}")

    if dataset_path.exists():

        data = pd.read_csv(dataset_path)

        print(f"現在のdataset.csvのデータ数: {len(data)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
