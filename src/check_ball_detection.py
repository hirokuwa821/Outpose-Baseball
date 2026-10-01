from ultralytics import YOLO
import cv2
from pathlib import Path
import csv

MODEL_PATH = "runs/detect/train-6/weights/best.pt"

VIDEO_PATH = Path("test_videos/mlb_082_88.8mph_BC50FOA5GINW.mp4")

OUTPUT_CSV = Path("output/mlb_082_detection_check.csv")

CONF = 0.20


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("動画を開けません")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 70)
print("YOLOボール検出チェック")
print("=" * 70)

print("動画:", VIDEO_PATH.name)
print("FPS:", fps)
print("総フレーム数:", total_frames)

detections = []

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(frame, conf=CONF, verbose=False)

    best = None

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            candidate = {
                "frame": frame_number,
                "confidence": confidence,
                "cx": cx,
                "cy": cy,
            }

            if best is None:
                best = candidate

            elif confidence > best["confidence"]:
                best = candidate

    if best is not None:

        detections.append(best)

print()
print("検出フレーム数:", len(detections))

if detections:

    first_frame = detections[0]["frame"]
    last_frame = detections[-1]["frame"]

    print("最初の検出:", first_frame)
    print("最後の検出:", last_frame)

    print("検出率:", f"{len(detections) / total_frames * 100:.2f}%")

    confidences = [d["confidence"] for d in detections]

    print("平均信頼度:", f"{sum(confidences) / len(confidences):.3f}")

    print("最低信頼度:", f"{min(confidences):.3f}")

    print("最高信頼度:", f"{max(confidences):.3f}")

else:

    print("ボールを1回も検出できませんでした")


OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow(["frame", "confidence", "center_x", "center_y"])

    for d in detections:

        writer.writerow([d["frame"], d["confidence"], d["cx"], d["cy"]])

cap.release()

print()
print("CSV保存:", OUTPUT_CSV)
print("=" * 70)
print("確認完了")
print("=" * 70)
