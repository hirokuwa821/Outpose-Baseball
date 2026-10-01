import cv2
from pathlib import Path

# ==========================================
# 設定
# ==========================================

VIDEOS = [
    Path("videos/pitch_102_1.mp4"),
    Path("videos/pitch_98_1.mp4"),
    Path("videos/pitch_98_2.mp4"),
]

OUTPUT_DIR = Path("ball_dataset/additional_images")

START_FRAME = 35
END_FRAME = 130

INTERVAL = 2


# ==========================================
# 出力フォルダ作成
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 既存画像を削除
# ==========================================

old_images = list(OUTPUT_DIR.glob("*.jpg"))

if old_images:

    print("既存画像を削除します:", len(old_images), "枚")

    for image in old_images:
        image.unlink()


# ==========================================
# 動画ごとに画像抽出
# ==========================================

total_saved = 0


for video_path in VIDEOS:

    print()
    print("=" * 60)
    print("動画:", video_path)
    print("=" * 60)

    if not video_path.exists():

        print("動画がありません:")
        print(video_path)

        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません")

        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("FPS:", fps)
    print("サイズ:", width, "x", height)
    print("総フレーム数:", total_frames)

    start = max(1, START_FRAME)
    end = min(END_FRAME, total_frames)

    print("抽出範囲:", start, "～", end)
    print("間隔:", INTERVAL, "フレーム")

    video_saved = 0

    # --------------------------------------
    # 動画名
    # --------------------------------------

    video_stem = video_path.stem

    # --------------------------------------
    # フレーム抽出
    # --------------------------------------

    for frame_number in range(start, end + 1, INTERVAL):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"{video_stem}" f"_frame_{frame_number:04d}.jpg"

        output_path = OUTPUT_DIR / filename

        cv2.imwrite(str(output_path), frame)

        video_saved += 1
        total_saved += 1

    cap.release()

    print("保存枚数:", video_saved)


# ==========================================
# 完了
# ==========================================

print()
print("=" * 60)
print("追加学習用画像抽出完了")
print("=" * 60)

print("総保存枚数:", total_saved)
print("保存先:", OUTPUT_DIR)

print()
print("画像ファイル例:")

images = sorted(OUTPUT_DIR.glob("*.jpg"))

for image in images[:15]:

    print(" ", image.name)

print()
print("==========================================")
print("確認してください")
print("==========================================")

print("画像数:", len(images))
