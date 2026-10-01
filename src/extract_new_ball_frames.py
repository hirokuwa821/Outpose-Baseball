import cv2
from pathlib import Path

VIDEO_PATH = "test_videos/mlb_002_81.9mph_MM9QJ9EPUPHF_right.mp4"

START_FRAME = 45
END_FRAME = 110

OUTPUT_DIR = Path("ball_dataset/new_pitch_images_4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

for frame_number in range(START_FRAME, END_FRAME + 1):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print("読み込み失敗:", frame_number)
        continue

    filename = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

    cv2.imwrite(str(filename), frame)

cap.release()

print("抽出完了")
print("保存先:", OUTPUT_DIR)
print("フレーム数:", END_FRAME - START_FRAME + 1)
