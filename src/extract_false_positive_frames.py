from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "runs/detect/train-6/weights/best.pt"

VIDEO_PATH = "test_videos/mlb_082_88.8mph_BC50FOA5GINW.mp4"

OUTPUT_DIR = Path("ball_dataset/false_positive_frames/mlb_082")

CONF = 0.01

# 投球付近だけ確認
START_FRAME = 140
END_FRAME = 165

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けませんでした")
    exit()

print("=" * 50)
print("誤検出候補フレームを抽出します")
print("フレーム:", START_FRAME, "～", END_FRAME)
print("Confidence:", CONF)
print("=" * 50)

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

frame_number = START_FRAME
saved_count = 0

while frame_number <= END_FRAME:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(source=frame, conf=CONF, verbose=False)

    result = results[0]

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, xyxy)

            detections.append((conf, x1, y1, x2, y2))

    # 検出があるフレームだけ保存
    if detections:

        detections.sort(reverse=True)

        # 全候補を描画
        for i, (conf, x1, y1, x2, y2) in enumerate(detections):

            if i == 0:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        filename = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

        cv2.imwrite(str(filename), frame)

        saved_count += 1

        print(
            f"保存: frame {frame_number} "
            f"(detections={len(detections)}, "
            f"best_conf={detections[0][0]:.3f})"
        )

    frame_number += 1

cap.release()

print()
print("=" * 50)
print("完了")
print("保存枚数:", saved_count)
print("保存先:", OUTPUT_DIR.resolve())
print("=" * 50)
