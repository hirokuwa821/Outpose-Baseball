import cv2
from pathlib import Path

# ============================================================
# 設定
# ============================================================

VIDEO_DIR = Path("videos")

OUTPUT_DIR = Path("ball_dataset/additional_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象動画
VIDEOS = [
    "pitch_102_1.mp4",
    "pitch_98_1.mp4",
    "pitch_98_2.mp4",
]

# 投球動作が含まれている可能性が高い範囲
# 3本とも少し広めに取る
START_FRAME = 35
END_FRAME = 130

# 何フレームおきに保存するか
# 2なら 35, 37, 39, ... のように保存
FRAME_INTERVAL = 2


# ============================================================
# 画像抽出
# ============================================================

total_saved = 0


for video_name in VIDEOS:

    video_path = VIDEO_DIR / video_name

    print()
    print("============================================================")
    print("動画:", video_path)
    print("============================================================")

    if not video_path.exists():

        print("動画が見つかりません:", video_path)
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("総フレーム数:", total_frames)

    start = max(1, START_FRAME)
    end = min(END_FRAME, total_frames)

    print("抽出範囲:", start, "～", end)
    print("間隔:", FRAME_INTERVAL, "フレーム")

    saved = 0

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # 範囲外
        if frame_number < start:
            continue

        if frame_number > end:
            break

        # 指定間隔だけ保存
        if (frame_number - start) % FRAME_INTERVAL != 0:
            continue

        # ファイル名
        stem = Path(video_name).stem

        output_name = f"{stem}_frame_{frame_number:04d}.jpg"

        output_path = OUTPUT_DIR / output_name

        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved += 1
        total_saved += 1

    cap.release()

    print("保存枚数:", saved)


# ============================================================
# 終了
# ============================================================

print()
print("============================================================")
print("追加学習用画像抽出完了")
print("============================================================")

print("総保存枚数:", total_saved)
print("保存先:", OUTPUT_DIR)
