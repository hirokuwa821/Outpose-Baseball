from ultralytics import YOLO
from pathlib import Path
import cv2

# ==========================================
# 設定
# ==========================================

MODEL_PATH = "runs/detect/runs/detect/" "ball_finetune_mydata_v2/weights/best.pt"

VIDEOS = [
    "videos/pitch_102_1.mp4",
    "videos/pitch_98_1.mp4",
    "videos/pitch_98_2.mp4",
]

OUTPUT_DIR = Path("runs/detect/finetuned_detection_test_v2")

CONF = 0.05

IMG_SIZE = 1280


# ==========================================
# モデル
# ==========================================

print("=" * 60)
print("追加学習モデル 検出テスト v2")
print("=" * 60)

print("モデル:")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print("モデル読み込み完了")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 動画ごとに検出
# ==========================================

for video_path in VIDEOS:

    print()
    print("#" * 60)
    print("動画:", video_path)
    print("#" * 60)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("動画を開けません")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("サイズ:", width, "x", height)
    print("総フレーム数:", total_frames)

    # --------------------------------------
    # 出力動画
    # --------------------------------------

    output_path = OUTPUT_DIR / f"{Path(video_path).stem}_detection.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # --------------------------------------
    # 検出情報
    # --------------------------------------

    detected_frames = []

    max_conf = 0.0
    max_conf_frame = None

    frame_number = 0

    # ======================================
    # 全フレーム処理
    # ======================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)

        best_detection = None

        # ==================================
        # 検出結果
        # ==================================

        for result in results:

            if result.boxes is None:
                continue

            if len(result.boxes) == 0:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if best_detection is None or confidence > best_detection["confidence"]:

                    best_detection = {
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }

        # ==================================
        # 検出あり
        # ==================================

        if best_detection is not None:

            confidence = best_detection["confidence"]

            x1 = best_detection["x1"]
            y1 = best_detection["y1"]
            x2 = best_detection["x2"]
            y2 = best_detection["y2"]

            cx = int((x1 + x2) / 2)

            cy = int((y1 + y2) / 2)

            detected_frames.append(frame_number)

            if confidence > max_conf:

                max_conf = confidence
                max_conf_frame = frame_number

            # ------------------------------
            # ボックス
            # ------------------------------

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ------------------------------
            # 中心点
            # ------------------------------

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # ------------------------------
            # 情報
            # ------------------------------

            cv2.putText(
                frame,
                f"BALL {confidence:.3f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"frame: {frame_number}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # ==================================
        # 検出なし
        # ==================================

        else:

            cv2.putText(
                frame,
                "NO BALL",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        writer.write(frame)

    cap.release()
    writer.release()

    # ======================================
    # 結果
    # ======================================

    detected_count = len(detected_frames)

    detection_rate = detected_count / total_frames * 100

    print()
    print("------------------------------------------")
    print("検出結果")
    print("------------------------------------------")

    print("検出フレーム数:", detected_count)

    print(f"検出率: {detection_rate:.1f}%")

    print(f"最高confidence: {max_conf:.4f}")

    print("最高confidenceフレーム:", max_conf_frame)

    print()
    print("検出フレーム:")

    print(detected_frames)

    print()
    print("結果動画:")
    print(output_path)


# ==========================================
# 終了
# ==========================================

print()
print("=" * 60)
print("全動画の検出テスト完了")
print("=" * 60)
