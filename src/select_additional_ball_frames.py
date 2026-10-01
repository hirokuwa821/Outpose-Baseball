import cv2
from pathlib import Path

# ==============================
# 追加ラベル用フレーム抽出
# ==============================

# 対象動画
VIDEO_PATH = "test_videos/mlb_001_81.1mph_CGI1SSOSP466_right.mp4"

# リリース前後を重点的に抽出
START_FRAME = 80
END_FRAME = 130

# 何フレームおきに保存するか
STEP = 2

# 保存先
OUTPUT_DIR = Path("ball_dataset/additional_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けませんでした:")
    print(VIDEO_PATH)
    raise SystemExit

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("総フレーム数:", total_frames)
print("FPS:", fps)
print(f"フレーム {START_FRAME}～{END_FRAME} を {STEP} フレームおきに抽出します。")

saved = 0

for frame_no in range(START_FRAME, min(END_FRAME + 1, total_frames), STEP):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    if not ret:
        print("読み込み失敗:", frame_no)
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_no:04d}.jpg"
    cv2.imwrite(str(output_path), frame)
    saved += 1

cap.release()

print("=" * 50)
print("抽出完了")
print("保存枚数:", saved)
print("保存先:", OUTPUT_DIR.resolve())
print("=" * 50)
print()
print("次に ball_label.py でこのフォルダの画像をラベル付けしてください。")
