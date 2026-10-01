import cv2
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 設定
# ==========================================

MODEL_PATH = Path("runs/detect/train-11/weights/best.pt")

VIDEO_DIR = Path("test_videos")

OUTPUT_DIR = Path("output/train11_3videos_check")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 確認する3動画
VIDEO_STEMS = [
    "mlb_135_90.0mph_328C7QXVBU0M",
    "mlb_140_",
    "mlb_142_",
]

# ==========================================
# モデル
# ==========================================

if not MODEL_PATH.exists():
    print("モデルがありません:")
    print(MODEL_PATH)
    raise SystemExit

model = YOLO(str(MODEL_PATH))

# ==========================================
# 動画検索
# ==========================================

all_videos = list(VIDEO_DIR.rglob("*.mp4"))

selected_videos = []

for stem in VIDEO_STEMS:

    matches = [p for p in all_videos if stem in p.stem]

    if matches:

        selected_videos.append(matches[0])

    else:

        print()
        print("動画が見つかりません:")
        print(stem)


if not selected_videos:

    print("確認できる動画がありません")
    raise SystemExit

# ==========================================
# 各動画を処理
# ==========================================

for video_path in selected_videos:

    print()
    print("=" * 60)
    print("処理中:")
    print(video_path.name)
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = OUTPUT_DIR / f"{video_path.stem}_train11_tracking.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detected_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model.predict(source=frame, conf=0.25, verbose=False)

        result = results[0]

        display = result.plot()

        if result.boxes is not None:
            count = len(result.boxes)
        else:
            count = 0

        if count > 0:
            detected_frames += 1

        cv2.putText(
            display, "train-11", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        writer.write(display)

    cap.release()
    writer.release()

    print()
    print("完了")
    print("全フレーム:", frame_count)
    print("検出フレーム:", detected_frames)
    print("保存先:")
    print(output_path)

print()
print("=" * 60)
print("3動画の確認動画作成完了")
print("=" * 60)
print("保存先:")
print(OUTPUT_DIR)
