import cv2
from pathlib import Path
from ultralytics import YOLO


VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

MODEL_PATH = "runs/detect/train-9/weights/best.pt"

OUTPUT_DIR = Path("output/mlb_007_train9_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 動画を探す
video_files = list(
    Path("test_videos").rglob(f"{VIDEO_STEM}.mp4")
)

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit


video_path = video_files[0]


# train-9モデル
model = YOLO(MODEL_PATH)


# 動画を読み込み
cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_path = (
    OUTPUT_DIR /
    f"{VIDEO_STEM}_train9_tracking.mp4"
)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)


frame_no = 0
detected_frames = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # train-9で検出
    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )


    result = results[0]


    if result.boxes is not None and len(result.boxes) > 0:

        detected_frames += 1


        # 最も信頼度の高いボール
        confidence = result.boxes.conf.cpu().numpy()
        best_index = confidence.argmax()


        box = result.boxes.xyxy[
            best_index
        ].cpu().numpy().astype(int)


        x1, y1, x2, y2 = box


        # ボールを赤枠で表示
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )


        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2


        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0, 255, 0),
            -1
        )


        cv2.putText(
            frame,
            f"Ball {confidence[best_index]:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


    # フレーム番号
    cv2.putText(
        frame,
        f"Frame {frame_no}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    writer.write(frame)

    frame_no += 1


cap.release()
writer.release()


print()
print("train-9での確認動画を作成しました")
print("検出フレーム数:", detected_frames)
print("全フレーム数:", frame_no)
print("保存先:", output_path)