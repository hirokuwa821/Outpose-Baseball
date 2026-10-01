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

OUTPUT_DIR = Path("runs/detect/finetuned_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.05

# ============================================================
# モデル
# ============================================================

print("==========================================")
print("追加学習モデル検証")
print("==========================================")

print("モデル:")
print(MODEL_PATH)

model = YOLO(MODEL_PATH)

print("モデル読み込み完了")

# ============================================================
# 各動画を検証
# ============================================================

for video_path in VIDEOS:

    print()
    print("==========================================")
    print("動画")
    print("==========================================")

    print(video_path)

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

    # --------------------------------------------------------
    # 出力動画
    # --------------------------------------------------------

    name = Path(video_path).stem

    output_path = OUTPUT_DIR / f"{name}_finetuned.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detection_frames = []
    max_conf = 0.0
    max_conf_frame = None

    frame_number = 0

    # --------------------------------------------------------
    # 全フレーム検出
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(frame, conf=CONF, verbose=False)

        candidates = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                candidates.append(
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

        # ----------------------------------------------------
        # 最高confidenceの検出
        # ----------------------------------------------------

        if candidates:

            best = max(candidates, key=lambda x: x["confidence"])

            detection_frames.append(frame_number)

            if best["confidence"] > max_conf:

                max_conf = best["confidence"]
                max_conf_frame = frame_number

            # ------------------------------------------------
            # 検出結果描画
            # ------------------------------------------------

            x1 = best["x1"]
            y1 = best["y1"]
            x2 = best["x2"]
            y2 = best["y2"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.circle(frame, (int(best["cx"]), int(best["cy"])), 4, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"BALL {best['confidence']:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

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

        # ----------------------------------------------------
        # フレーム番号
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"frame: {frame_number}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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

    if detection_frames:

        print("最初:", detection_frames[0])

        print("最後:", detection_frames[-1])

        print("最高confidence:", f"{max_conf:.4f}")

        print("最高confidenceフレーム:", max_conf_frame)

        print()
        print("検出フレーム:")
        print(detection_frames)

    else:

        print("ボールを検出できませんでした")

    print()
    print("結果動画:")
    print(output_path)

print()
print("==========================================")
print("全動画の検証完了")
print("==========================================")
