from pathlib import Path
import cv2
from ultralytics import YOLO
import pandas as pd

# ==============================
# 設定
# ==============================

MODEL_PATH = "runs/detect/train-5/weights/best.pt"

VIDEO_LIST = "output/ball_label_candidates_20.csv"

OUTPUT_DIR = Path("ball_dataset/auto_candidates")

CONF = 0.20

# 検出があった前後何フレームを保存するか
FRAME_MARGIN = 3


# ==============================
# モデル読み込み
# ==============================

model = YOLO(MODEL_PATH)

videos = pd.read_csv(VIDEO_LIST)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 動画ごとに処理
# ==============================

for video_name in videos["video"]:

    video_path = Path("test_videos") / video_name

    print()
    print("=" * 60)
    print("解析中:", video_name)
    print("=" * 60)

    if not video_path.exists():
        print("動画が見つかりません:", video_path)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("動画を開けません:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("総フレーム:", total_frames)

    detections = []

    frame_index = 0

    # ==============================
    # 全フレームを解析
    # ==============================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=CONF, verbose=False)

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:

            # 一番confidenceが高い検出を使用
            best_index = int(result.boxes.conf.argmax())

            confidence = float(result.boxes.conf[best_index])

            box = result.boxes.xyxy[best_index].cpu().numpy()

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append(
                {
                    "frame": frame_index,
                    "confidence": confidence,
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

        frame_index += 1

    cap.release()

    print("検出フレーム数:", len(detections))

    if len(detections) == 0:
        print("検出なし")
        continue

    # ==============================
    # 検出フレームをまとめる
    # ==============================

    detection_frames = [d["frame"] for d in detections]

    candidate_frames = set()

    for frame_number in detection_frames:

        for offset in range(-FRAME_MARGIN, FRAME_MARGIN + 1):

            target = frame_number + offset

            if 0 <= target < total_frames:
                candidate_frames.add(target)

    candidate_frames = sorted(candidate_frames)

    print("候補フレーム数:", len(candidate_frames))

    # ==============================
    # 出力フォルダ
    # ==============================

    video_stem = video_path.stem

    output_dir = OUTPUT_DIR / video_stem

    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 候補フレーム保存
    # ==============================

    cap = cv2.VideoCapture(str(video_path))

    frame_index = 0

    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index in candidate_frames:

            output_path = output_dir / f"frame_{frame_index:05d}.jpg"

            cv2.imwrite(str(output_path), frame)

            saved += 1

        frame_index += 1

    cap.release()

    print("保存枚数:", saved)

print()
print("=" * 60)
print("20本の候補フレーム抽出完了")
print("=" * 60)
