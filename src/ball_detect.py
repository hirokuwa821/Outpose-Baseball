from ultralytics import YOLO
from pathlib import Path
import cv2

# ボール検出用モデル
model = YOLO("yolo11n.pt")

# 動画
video_path = Path("test_videos/mlb_001_81.1mph_CGI1SSOSP466_right.mp4")

# 出力動画
output_path = Path("output/ball_detect.mp4")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("サイズ:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

frame_count = 0
detections = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLOで検出
    results = model(frame, verbose=False)

    # 検出結果を描画
    annotated = results[0].plot()

    # sports ball = COCO class 32
    for box, cls, conf in zip(
        results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
    ):
        if int(cls) == 32 and float(conf) >= 0.1:
            detections += 1

            x1, y1, x2, y2 = map(int, box)

            print(
                f"frame={frame_count}, "
                f"ball=({(x1+x2)//2}, {(y1+y2)//2}), "
                f"conf={float(conf):.3f}"
            )

    out.write(annotated)

    frame_count += 1

cap.release()
out.release()

print()
print("解析終了")
print("フレーム数:", frame_count)
print("ボール検出数:", detections)
print("出力:", output_path)
