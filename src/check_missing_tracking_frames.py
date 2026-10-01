import cv2
from pathlib import Path

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

# 元動画を探す
video_files = list(Path("test_videos").rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

video_path = video_files[0]

# 確認するフレーム
target_frames = [120, 125, 130, 135, 140, 142]

# 保存先
output_dir = Path("output/mlb_007_missing_frames")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

for frame_no in target_frames:

    # 指定フレームへ移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if not ret:
        print(f"Frame {frame_no}: 読み込み失敗")
        continue

    # 保存
    output_path = output_dir / f"mlb_007_frame_{frame_no}.jpg"

    cv2.imwrite(str(output_path), frame)

    print(f"保存: {output_path}")

cap.release()

print("フレーム画像の作成が完了しました")
