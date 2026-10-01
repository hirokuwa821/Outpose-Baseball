import cv2
import math
import zipfile
from pathlib import Path
import numpy as np

# ==========================================
# 設定
# ==========================================

VIDEO_PATH = "videos/pitch_102_1.mp4"

# 投球付近のフレーム範囲
START_FRAME = 45
END_FRAME = 105

# 何フレームおきに抽出するか
FRAME_STEP = 2

# 出力先
OUTPUT_DIR = Path("output/pitch_102_annotation_frames")

CONTACT_SHEET = Path("output/pitch_102_annotation_contact_sheet.jpg")

ZIP_PATH = Path("output/pitch_102_annotation_frames.zip")


# ==========================================
# 出力フォルダ作成
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# 動画確認
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("動画を開けません:")
    print(VIDEO_PATH)

    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print("==========================================")
print("投球フレーム抽出")
print("==========================================")

print("動画:", VIDEO_PATH)
print("FPS:", fps)
print("サイズ:", width, "x", height)
print("総フレーム数:", total_frames)


# ==========================================
# フレーム範囲調整
# ==========================================

start_frame = max(1, START_FRAME)

end_frame = min(total_frames, END_FRAME)


print()
print("抽出範囲:")
print(f"フレーム {start_frame} ～ {end_frame}")

print(f"{FRAME_STEP}フレームおきに抽出")


# ==========================================
# フレーム抽出
# ==========================================

saved_frames = []


for frame_number in range(start_frame, end_frame + 1, FRAME_STEP):

    # 指定フレームへ移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, frame = cap.read()

    if not ret:

        print(f"フレーム {frame_number} " "を読み込めません")

        continue

    filename = f"frame_{frame_number:04d}.jpg"

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    saved_frames.append((frame_number, output_path))


cap.release()


# ==========================================
# 抽出結果
# ==========================================

print()
print("==========================================")
print("抽出完了")
print("==========================================")

print("抽出枚数:", len(saved_frames))

print("保存先:", OUTPUT_DIR)


# ==========================================
# コンタクトシート
# ==========================================

if len(saved_frames) == 0:

    print("画像がありません")
    exit()


# サムネイルサイズ
THUMB_WIDTH = 180
THUMB_HEIGHT = 320

# 1行あたりの画像数
COLS = 6

# フレーム番号表示用スペース
LABEL_HEIGHT = 35


rows = math.ceil(len(saved_frames) / COLS)


sheet_width = COLS * THUMB_WIDTH

sheet_height = rows * (THUMB_HEIGHT + LABEL_HEIGHT)


sheet = np.ones((sheet_height, sheet_width, 3), dtype=np.uint8) * 255


# ==========================================
# コンタクトシート作成
# ==========================================

for i, (frame_number, image_path) in enumerate(saved_frames):

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.resize(image, (THUMB_WIDTH, THUMB_HEIGHT))

    row = i // COLS
    col = i % COLS

    x = col * THUMB_WIDTH
    y = row * (THUMB_HEIGHT + LABEL_HEIGHT)

    sheet[y : y + THUMB_HEIGHT, x : x + THUMB_WIDTH] = image

    # フレーム番号
    cv2.putText(
        sheet,
        f"Frame {frame_number}",
        (x + 5, y + THUMB_HEIGHT + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


# ==========================================
# コンタクトシート保存
# ==========================================

cv2.imwrite(str(CONTACT_SHEET), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


print()
print("コンタクトシート:")
print(CONTACT_SHEET)


# ==========================================
# ZIP作成
# ==========================================

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zip_file:

    for frame_number, image_path in saved_frames:

        zip_file.write(image_path, arcname=image_path.name)


print()
print("ZIP:")
print(ZIP_PATH)


# ==========================================
# 終了
# ==========================================

print()
print("==========================================")
print("処理完了")
print("==========================================")

print()
print("作成ファイル:")

print("個別画像:", OUTPUT_DIR)

print("一覧画像:", CONTACT_SHEET)

print("ZIP:", ZIP_PATH)
