import cv2
from pathlib import Path

# ============================================================
# 設定
# ============================================================

VIDEO_PATH = "videos/pitch_102_1.mp4"

OUTPUT_DIR = Path("dataset/ball_annotation/images")

# 投球付近のフレーム範囲
START_FRAME = 35
END_FRAME = 80

# 何フレームごとに保存するか
FRAME_INTERVAL = 2


# ============================================================
# 出力フォルダ
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 動画
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("動画を開けません")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print()
print("==========================================")
print("ボールアノテーション用画像作成")
print("==========================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)

print("抽出範囲:", START_FRAME, "～", END_FRAME)

print("間隔:", FRAME_INTERVAL, "フレーム")


# ============================================================
# フレーム抽出
# ============================================================

frame_number = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # 範囲外
    if frame_number < START_FRAME:
        continue

    if frame_number > END_FRAME:
        break

    # 指定間隔
    if (frame_number - START_FRAME) % FRAME_INTERVAL != 0:
        continue

    # ========================================================
    # 保存
    # ========================================================

    filename = f"pitch_102_frame_{frame_number:04d}.jpg"

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(str(output_path), frame)

    saved_count += 1

    print(f"保存: frame {frame_number:3d} -> {output_path}")


# ============================================================
# 終了
# ============================================================

cap.release()


print()
print("==========================================")
print("完了")
print("==========================================")

print("保存枚数:", saved_count)

print("保存先:", OUTPUT_DIR)
