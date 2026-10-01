import cv2
import pandas as pd
from pathlib import Path

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"
CSV_PATH = Path("output/ball_tracking_v2") / f"{VIDEO_STEM}_tracking.csv"

# 動画を探す
video_files = list(Path("test_videos").rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

VIDEO_PATH = video_files[0]
OUTPUT_PATH = Path("output") / f"{VIDEO_STEM}_tracking_check.mp4"

df = pd.read_csv(CSV_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

tracking = {
    int(row.frame): (int(row.center_x), int(row.center_y), float(row.confidence))
    for _, row in df.iterrows()
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    if frame_no in tracking:
        x, y, conf = tracking[frame_no]

        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Ball conf: {conf:.2f}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    out.write(frame)

cap.release()
out.release()

print("確認動画を作成しました:")
print(OUTPUT_PATH)
