import cv2
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 設定
# ==========================================

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

MODEL8 = Path("runs/detect/train-8/weights/best.pt")
MODEL10 = Path("runs/detect/train-10/weights/best.pt")

OUTPUT_DIR = Path("output/train8_train10_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = list(Path("test_videos").rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

video_path = video_files[0]

# ==========================================
# モデル読み込み
# ==========================================

print("train-8 読み込み...")
model8 = YOLO(str(MODEL8))

print("train-10 読み込み...")
model10 = YOLO(str(MODEL10))

# ==========================================
# 動画
# ==========================================

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("動画を開けません")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ==========================================
# 出力
# ==========================================

output_path = OUTPUT_DIR / f"{VIDEO_STEM}_train8_vs_train10.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width * 2, height),
)

# ==========================================
# 推論
# ==========================================

frame_count = 0

train8_detected = 0
train10_detected = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ------------------------------
    # train-8
    # ------------------------------

    result8 = model8.predict(
        source=frame,
        conf=0.25,
        verbose=False,
    )[0]

    # ------------------------------
    # train-10
    # ------------------------------

    result10 = model10.predict(
        source=frame,
        conf=0.25,
        verbose=False,
    )[0]

    # ------------------------------
    # 描画
    # ------------------------------

    display8 = result8.plot()
    display10 = result10.plot()

    # ------------------------------
    # 検出数
    # ------------------------------

    count8 = 0
    count10 = 0

    if result8.boxes is not None:
        count8 = len(result8.boxes)

    if result10.boxes is not None:
        count10 = len(result10.boxes)

    if count8 > 0:
        train8_detected += 1

    if count10 > 0:
        train10_detected += 1

    # ------------------------------
    # モデル名を表示
    # ------------------------------

    cv2.putText(
        display8,
        "train-8",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display10,
        "train-10",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    # ------------------------------
    # 横に結合
    # ------------------------------

    combined = cv2.hconcat([display8, display10])

    writer.write(combined)

    if frame_count % 20 == 0:

        print(
            f"{frame_count}/{total_frames} | "
            f"train-8検出: {train8_detected} | "
            f"train-10検出: {train10_detected}"
        )

# ==========================================
# 終了
# ==========================================

cap.release()
writer.release()

print()
print("=" * 60)
print("比較動画を作成しました")
print("=" * 60)

print("全フレーム:", frame_count)
print("train-8 検出フレーム:", train8_detected)
print("train-10 検出フレーム:", train10_detected)
print()
print("保存先:")
print(output_path)
