from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ==========================================
# 設定
# ==========================================

MODEL_PATH = Path("runs/detect/train-11/weights/best.pt")

VIDEO_DIR = Path("test_videos")

OUTPUT_DIR = Path("output/train11_tracking_all")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# モデル確認
# ==========================================

if not MODEL_PATH.exists():
    print("モデルが見つかりません:")
    print(MODEL_PATH)
    raise SystemExit

model = YOLO(str(MODEL_PATH))


# ==========================================
# 動画取得
# ==========================================

videos = sorted(VIDEO_DIR.rglob("*.mp4"))

if not videos:
    print("動画が見つかりません")
    raise SystemExit


print("=" * 60)
print("train-11 全動画追跡")
print("=" * 60)

print("モデル:", MODEL_PATH)
print("動画数:", len(videos))
print()


# ==========================================
# 全動画を処理
# ==========================================

summary = []

for video_index, video_path in enumerate(videos, start=1):

    print("=" * 60)
    print(f"[{video_index}/{len(videos)}] " f"{video_path.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("動画を開けません")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []

    frame_no = 0
    detected_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ==================================
        # train-11で検出
        # ==================================

        results = model.predict(source=frame, conf=0.25, verbose=False)

        result = results[0]

        # ==================================
        # 検出結果
        # ==================================

        if result.boxes is not None and len(result.boxes) > 0:

            # 信頼度が最大の検出を使用
            confidences = result.boxes.conf.cpu().numpy()

            best_index = confidences.argmax()

            xyxy = result.boxes.xyxy[best_index].cpu().numpy()

            confidence = float(confidences[best_index])

            x1, y1, x2, y2 = xyxy

            center_x = (x1 + x2) / 2

            center_y = (y1 + y2) / 2

            detected_frames += 1

            rows.append(
                {
                    "frame": frame_no,
                    "time_sec": (frame_no / fps if fps > 0 else 0),
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": confidence,
                }
            )

        frame_no += 1

        # ==================================
        # 進捗表示
        # ==================================

        if frame_no % 50 == 0:

            print(f"{frame_no}/{total_frames} " f"| 検出 {detected_frames}")

    cap.release()

    # ======================================
    # CSV保存
    # ======================================

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "time_sec",
            "center_x",
            "center_y",
            "confidence",
        ],
    )

    output_path = OUTPUT_DIR / f"{video_path.stem}_tracking.csv"

    df.to_csv(output_path, index=False)

    # ======================================
    # 結果
    # ======================================

    detection_rate = detected_frames / frame_no if frame_no > 0 else 0

    print()
    print("完了")
    print("全フレーム:", frame_no)
    print("検出フレーム:", detected_frames)
    print("検出率:", f"{detection_rate * 100:.1f}%")
    print("保存:", output_path)

    summary.append(
        {
            "video": video_path.name,
            "frames": frame_no,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
        }
    )


# ==========================================
# 全動画サマリー
# ==========================================

summary_df = pd.DataFrame(summary)

summary_path = OUTPUT_DIR / "train11_tracking_summary.csv"

summary_df.to_csv(summary_path, index=False)


print()
print("=" * 60)
print("全動画の追跡完了")
print("=" * 60)

print("処理動画数:", len(summary_df))

print("結果保存先:", OUTPUT_DIR)

print("サマリー:", summary_path)
