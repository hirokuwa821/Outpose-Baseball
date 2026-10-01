import cv2
from pathlib import Path

VIDEO_PATH = "test_videos/mlb_001_81.5mph_KPY8PN4UY84W_right.mp4"

START_FRAME = 50
END_FRAME = 100

OUTPUT_DIR = Path("ball_dataset/pitch2/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()

for frame_number in range(START_FRAME, END_FRAME + 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print("読み込み失敗:", frame_number)
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
    cv2.imwrite(str(output_path), frame)

cap.release()

print()
print("フレーム抽出完了")
print("保存先:", OUTPUT_DIR)
