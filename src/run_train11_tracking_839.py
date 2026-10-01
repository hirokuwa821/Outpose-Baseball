from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ============================================================
# 設定
# ============================================================

MODEL_PATH = Path("runs/detect/train-11/weights/best.pt")

VIDEO_LIST = Path("output/unique_videos_with_speed.csv")

VIDEO_DIR = Path("test_videos")

OUTPUT_DIR = Path("output/train11_tracking_839")


# ============================================================
# 出力フォルダ作成
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ファイル確認
# ============================================================

if not MODEL_PATH.exists():
    print("モデルが見つかりません:")
    print(MODEL_PATH)
    raise SystemExit


if not VIDEO_LIST.exists():
    print("動画リストが見つかりません:")
    print(VIDEO_LIST)
    raise SystemExit


if not VIDEO_DIR.exists():
    print("動画フォルダが見つかりません:")
    print(VIDEO_DIR)
    raise SystemExit


# ============================================================
# モデル読み込み
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# 動画リスト読み込み
# ============================================================

video_df = pd.read_csv(VIDEO_LIST)

if "video" not in video_df.columns:
    print("動画リストに video 列がありません")
    raise SystemExit


video_names = video_df["video"].dropna().astype(str).tolist()


# ============================================================
# 実際に存在する動画だけ確認
# ============================================================

videos = []

missing_videos = []

for video_name in video_names:

    video_path = VIDEO_DIR / video_name

    if video_path.exists():
        videos.append(video_path)

    else:
        missing_videos.append(video_name)


# ============================================================
# 開始表示
# ============================================================

print("=" * 60)
print("train-11 839本 ボール追跡")
print("=" * 60)

print()
print("モデル:")
print(MODEL_PATH)

print()
print("動画リスト:")
print(VIDEO_LIST)

print()
print("リスト上の動画数:", len(video_names))

print("実際に存在する動画:", len(videos))

print("見つからない動画:", len(missing_videos))

print()
print("出力先:")
print(OUTPUT_DIR)


# ============================================================
# 見つからない動画
# ============================================================

if missing_videos:

    print()
    print("見つからない動画:")

    for name in missing_videos[:20]:
        print(" -", name)

    if len(missing_videos) > 20:
        print(f" ... その他 {len(missing_videos) - 20} 本")


# ============================================================
# 追跡
# ============================================================

summary = []

success_count = 0

skip_count = 0

fail_count = 0


for video_index, video_path in enumerate(videos, start=1):

    print()
    print("=" * 60)

    print(f"[{video_index}/{len(videos)}] " f"{video_path.name}")

    print("=" * 60)

    # --------------------------------------------------------
    # 出力CSV
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / f"{video_path.stem}_tracking.csv"

    # --------------------------------------------------------
    # 既に処理済みならスキップ
    # --------------------------------------------------------

    if output_path.exists():

        print("既に処理済み → スキップ")

        skip_count += 1

        continue

    # --------------------------------------------------------
    # 動画を開く
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けませんでした")

        fail_count += 1

        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")

    print(f"総フレーム数: {total_frames}")

    # --------------------------------------------------------
    # 結果格納
    # --------------------------------------------------------

    rows = []

    frame_no = 0

    detected_frames = 0

    # --------------------------------------------------------
    # 全フレーム処理
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # train-11モデルによる検出
        # ----------------------------------------------------

        results = model.predict(source=frame, conf=0.25, verbose=False)

        result = results[0]

        # ----------------------------------------------------
        # 検出結果
        # ----------------------------------------------------

        if result.boxes is not None and len(result.boxes) > 0:

            confidences = result.boxes.conf.cpu().numpy()

            # 最もconfidenceが高い検出を使用

            best_index = confidences.argmax()

            xyxy = result.boxes.xyxy[best_index].cpu().numpy()

            confidence = float(confidences[best_index])

            x1, y1, x2, y2 = xyxy

            center_x = (float(x1) + float(x2)) / 2.0

            center_y = (float(y1) + float(y2)) / 2.0

            rows.append(
                {
                    "frame": frame_no,
                    "time_sec": (frame_no / fps if fps > 0 else 0),
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": confidence,
                }
            )

            detected_frames += 1

        # ----------------------------------------------------
        # 進捗表示
        # ----------------------------------------------------

        frame_no += 1

        if frame_no % 50 == 0:

            print(
                f"{frame_no}/{total_frames} " f"| 検出フレーム: " f"{detected_frames}"
            )

    # --------------------------------------------------------
    # 動画終了
    # --------------------------------------------------------

    cap.release()

    # --------------------------------------------------------
    # CSV保存
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "time_sec",
            "center_x",
            "center_y",
            "confidence",
        ],
    )

    result_df.to_csv(output_path, index=False)

    # --------------------------------------------------------
    # 検出率
    # --------------------------------------------------------

    detection_rate = detected_frames / frame_no if frame_no > 0 else 0

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()

    print("処理完了")

    print("全フレーム:", frame_no)

    print("検出フレーム:", detected_frames)

    print("検出率:", f"{detection_rate * 100:.1f}%")

    print("保存:", output_path)

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary.append(
        {
            "video": video_path.name,
            "frames": frame_no,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
        }
    )

    success_count += 1


# ============================================================
# summary保存
# ============================================================

summary_df = pd.DataFrame(summary)


summary_path = OUTPUT_DIR / "train11_tracking_summary.csv"


summary_df.to_csv(summary_path, index=False)


# ============================================================
# 最終結果
# ============================================================

print()
print("=" * 60)
print("839本 追跡処理完了")
print("=" * 60)

print()

print("対象動画:", len(videos))

print("追跡成功:", success_count)

print("スキップ:", skip_count)

print("失敗:", fail_count)

print()

print("作成CSV数:", len(list(OUTPUT_DIR.glob("*_tracking.csv"))))

print()

print("保存先:")

print(OUTPUT_DIR)

print()

print("サマリー:")

print(summary_path)

print()
print("=" * 60)
print("処理終了")
print("=" * 60)
