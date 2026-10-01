import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

MODEL_PATH = Path("runs/detect/train-11/weights/best.pt")

OUTPUT_DIR = Path("output/mlb_007_train11_check")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = list(Path("test_videos").rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

video_path = video_files[0]

print("モデル:", MODEL_PATH)
print("動画:", video_path)

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("動画を開けません")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = OUTPUT_DIR / f"{VIDEO_STEM}_train11_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

frame_count = 0
detected_frames = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model.predict(source=frame, conf=0.25, verbose=False)

    result = results[0]

    display = result.plot()

    if result.boxes is not None:
        count = len(result.boxes)
    else:
        count = 0

    if count > 0:
        detected_frames += 1

    cv2.putText(
        display, "train-11", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    writer.write(display)

    if frame_count % 20 == 0:

        print(f"{frame_count}/{total_frames} " f"| 検出フレーム: {detected_frames}")

cap.release()
writer.release()

print()
print("=" * 50)
print("train-11確認動画を作成しました")
print("=" * 50)

print("検出フレーム数:", detected_frames)
print("全フレーム数:", frame_count)
print("保存先:", output_path)
