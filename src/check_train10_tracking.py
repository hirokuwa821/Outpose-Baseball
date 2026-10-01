import cv2
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 設定
# ==========================================

VIDEO_STEM = "mlb_007_84.1mph_0JMTIRJHGH8V_right"

MODEL_PATH = Path("runs/detect/train-10/weights/best.pt")

OUTPUT_DIR = Path("output/mlb_007_train10_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 元動画を探す
video_files = list(Path("test_videos").rglob(f"{VIDEO_STEM}.mp4"))

if not video_files:
    print("元動画が見つかりません")
    raise SystemExit

video_path = video_files[0]

# ==========================================
# モデル読み込み
# ==========================================

print("モデル:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

# ==========================================
# 動画読み込み
# ==========================================

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("動画を開けません")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("動画:", video_path)
print("解像度:", width, "x", height)
print("FPS:", fps)
print("全フレーム数:", total_frames)

# ==========================================
# 出力設定
# ==========================================

output_path = OUTPUT_DIR / f"{VIDEO_STEM}_train10_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height),
)

# ==========================================
# 推論
# ==========================================

frame_count = 0
detected_frames = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    # 検出結果を描画
    annotated = result.plot()

    # 検出数
    if result.boxes is not None:
        num_boxes = len(result.boxes)

        if num_boxes > 0:
            detected_frames += 1

    writer.write(annotated)

    # 進捗表示
    if frame_count % 20 == 0:
        print(
            f"{frame_count}/{total_frames}",
            f"検出フレーム: {detected_frames}",
        )

# ==========================================
# 終了
# ==========================================

cap.release()
writer.release()

print()
print("=" * 60)
print("train-10 確認動画を作成しました")
print("=" * 60)
print("検出フレーム数:", detected_frames)
print("全フレーム数:", frame_count)
print("保存先:", output_path)
