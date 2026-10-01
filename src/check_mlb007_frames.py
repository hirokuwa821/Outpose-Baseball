import cv2
from pathlib import Path

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("output/mlb_007_frames_130_150")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = list(VIDEO_DIR.rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

video_path = video_files[0]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("動画を開けません")
    raise SystemExit

target_frames = range(130, 151)

for frame_no in target_frames:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if not ret:
        print(f"Frame {frame_no}: 読み込み失敗")
        continue

    output_path = OUTPUT_DIR / f"mlb_007_frame_{frame_no:03d}.jpg"

    cv2.imwrite(
        str(output_path),
        frame,
    )

    print(f"保存: {output_path}")

cap.release()

print()
print("フレーム画像の作成が完了しました")
print("保存先:", OUTPUT_DIR)
