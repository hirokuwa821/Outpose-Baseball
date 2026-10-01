from ultralytics import YOLO
import cv2
from pathlib import Path

VIDEO = "test_videos/mlb_007_84.1mph_0JMTIRJHGH8V_right.mp4"
FRAME_NO = 140

models = {
    "train8": "runs/detect/train-8/weights/best.pt",
    "train9": "runs/detect/train-9/weights/best.pt",
}

output_dir = Path("output/train8_train9_frame140")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Frame 140を読み込めませんでした")
    raise SystemExit

for name, model_path in models.items():
    model = YOLO(model_path)

    results = model.predict(frame, conf=0.25, verbose=False)

    result = results[0]
    output = frame.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(
                output,
                f"Ball {conf:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    save_path = output_dir / f"{name}_frame140.jpg"
    cv2.imwrite(str(save_path), output)

    print(f"{name}: {len(result.boxes)} detections")
    print(f"保存: {save_path}")
