from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "runs/detect/train-6/weights/best.pt"

VIDEO_PATH = "test_videos/mlb_082_88.8mph_BC50FOA5GINW.mp4"

OUTPUT_DIR = Path("ball_dataset/additional_frames/mlb_082")

CONF = 0.01

START_FRAME = 140
END_FRAME = 165

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

saved = 0

for frame_number in range(START_FRAME, END_FRAME + 1):

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(source=frame, conf=CONF, verbose=False)

    result = results[0]

    # YOLOが何かを検出したフレームだけ保存
    if result.boxes is not None and len(result.boxes) > 0:

        output_path = OUTPUT_DIR / (f"frame_{frame_number:04d}.jpg")

        # 枠や文字を一切描かず、
        # 元の画像そのままを保存
        cv2.imwrite(str(output_path), frame)

        saved += 1

        print(f"保存: frame {frame_number}")

cap.release()

print()
print("=" * 50)
print("完了")
print("保存枚数:", saved)
print("保存先:", OUTPUT_DIR.resolve())
print("=" * 50)

