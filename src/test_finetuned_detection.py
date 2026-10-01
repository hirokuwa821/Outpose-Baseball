from ultralytics import YOLO
import cv2
from pathlib import Path

# ============================================================
# 設定
# ============================================================

MODEL_PATH = "runs/detect/runs/detect/" "ball_finetune_mydata/weights/best.pt"

VIDEOS = [
    "videos/pitch_102_1.mp4",
    "videos/pitch_98_1.mp4",
    "videos/pitch_98_2.mp4",
]

OUTPUT_DIR = Path("runs/detect/finetuned_detection_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# まずは低め
CONF = 0.10


# ============================================================
# モデル
# ============================================================

print("==========================================")
print("追加学習モデル 検出テスト")
print("==========================================")

print("モデル:")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print("モデル読み込み完了")


# ============================================================
# 動画ごと
# ============================================================

for video_path in VIDEOS:

    print()
    print("############################################################")
    print("動画:", video_path)
    print("############################################################")

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

    video_name = Path(video_path).stem

    output_path = OUTPUT_DIR / f"{video_name}_detection.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = 0
    detection_frames = []
    all_detections = []

    max_conf = 0.0
    max_conf_frame = None

    # ========================================================
    # 全フレーム検出
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(frame, conf=CONF, verbose=False)

        frame_detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                frame_detections.append(
                    {
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                    }
                )

                all_detections.append(
                    {
                        "frame": frame_number,
                        "confidence": confidence,
                        "cx": cx,
                        "cy": cy,
                    }
                )

        # ====================================================
        # 検出あり
        # ====================================================

        if frame_detections:

            detection_frames.append(frame_number)

            # confidenceの高い順
            frame_detections.sort(key=lambda x: x["confidence"], reverse=True)

            # 一番confidenceが高いもの
            best = frame_detections[0]

            if best["confidence"] > max_conf:

                max_conf = best["confidence"]
                max_conf_frame = frame_number

            # ------------------------------------------------
            # 全候補を描画
            # ------------------------------------------------

            for i, d in enumerate(frame_detections):

                x1 = d["x1"]
                y1 = d["y1"]
                x2 = d["x2"]
                y2 = d["y2"]

                # 一番confidenceが高いものは太くする
                thickness = 4 if i == 0 else 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                cv2.putText(
                    frame,
                    f"{d['confidence']:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # ------------------------------------------------
            # 情報表示
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"DETECTIONS: {len(frame_detections)}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"BEST: {best['confidence']:.3f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # ====================================================
        # 検出なし
        # ====================================================

        else:

            cv2.putText(
                frame,
                "NO DETECTION",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        # ====================================================
        # フレーム番号
        # ====================================================

        cv2.putText(
            frame,
            f"FRAME: {frame_number}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    # ========================================================
    # 結果
    # ========================================================

    print()
    print("------------------------------------------")
    print("検出結果")
    print("------------------------------------------")

    print("検出フレーム数:", len(detection_frames))

    print("検出率:", f"{len(detection_frames) / total_frames * 100:.1f}%")

    if max_conf_frame is not None:

        print("最高confidence:", f"{max_conf:.4f}")

        print("最高confidenceフレーム:", max_conf_frame)

    print()
    print("検出フレーム:")
    print(detection_frames)

    print()
    print("結果動画:")
    print(output_path)


print()
print("==========================================")
print("全動画の検出テスト完了")
print("==========================================")
