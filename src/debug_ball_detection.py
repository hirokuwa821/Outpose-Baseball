from ultralytics import YOLO
import cv2
import sys
from pathlib import Path

MODEL_PATH = "runs/detect/train-2/weights/best.pt"
CONF = 0.05


if len(sys.argv) < 2:
    print("使い方:")
    print("python src/debug_ball_detection.py videos\\pitch_102_1.mp4")
    sys.exit(1)


VIDEO_PATH = sys.argv[1]

if not Path(VIDEO_PATH).exists():
    print("動画がありません:")
    print(VIDEO_PATH)
    sys.exit(1)


print("==========================================")
print("ボール検出デバッグ")
print("==========================================")
print("動画:", VIDEO_PATH)
print("モデル:", MODEL_PATH)
print("CONF:", CONF)


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    sys.exit(1)


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print()
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)


detections = []

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(frame, conf=CONF, verbose=False)

    frame_candidates = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            frame_candidates.append(
                {
                    "confidence": confidence,
                    "cx": cx,
                    "cy": cy,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    if frame_candidates:

        best = max(frame_candidates, key=lambda x: x["confidence"])

        detections.append({"frame": frame_number, **best})

        print(
            f"frame {frame_number:3d} | "
            f"detections={len(frame_candidates):2d} | "
            f"conf={best['confidence']:.3f} | "
            f"center=({best['cx']},{best['cy']})"
        )


cap.release()


print()
print("==========================================")
print("検出結果")
print("==========================================")

print("総フレーム数:", total_frames)
print("検出フレーム数:", len(detections))


if detections:

    print()
    print("最初の検出:")
    print(detections[0])

    print()
    print("最後の検出:")
    print(detections[-1])

    print()
    print("検出フレーム一覧:")

    print([d["frame"] for d in detections])

else:

    print()
    print("ボールを1フレームも検出できませんでした。")
