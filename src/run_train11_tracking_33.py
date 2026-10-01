from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO


# ==========================================
# 設定
# ==========================================

MODEL_PATH = Path(
    "runs/detect/train-11/weights/best.pt"
)

VIDEO_DIR = Path("test_videos")

VIDEO_LIST = Path(
    "output/ball_tracking_quality.csv"
)

OUTPUT_DIR = Path(
    "output/train11_tracking_33"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# モデル確認
# ==========================================

if not MODEL_PATH.exists():
    print("モデルが見つかりません:")
    print(MODEL_PATH)
    raise SystemExit


# ==========================================
# 33動画リスト確認
# ==========================================

if not VIDEO_LIST.exists():
    print("動画リストが見つかりません:")
    print(VIDEO_LIST)
    raise SystemExit


quality_df = pd.read_csv(
    VIDEO_LIST
)

if "video" not in quality_df.columns:
    print("video列がありません")
    print(quality_df.columns.tolist())
    raise SystemExit


video_names = (
    quality_df["video"]
    .dropna()
    .astype(str)
    .tolist()
)


print("=" * 60)
print("train-11 33動画 再追跡")
print("=" * 60)

print("リスト:", VIDEO_LIST)
print("動画数:", len(video_names))
print()


if len(video_names) != 33:
    print("警告: 動画数が33ではありません")
    print("動画数:", len(video_names))
    raise SystemExit


# ==========================================
# モデル読み込み
# ==========================================

model = YOLO(
    str(MODEL_PATH)
)


# ==========================================
# 動画処理
# ==========================================

summary = []

for index, tracking_name in enumerate(
    video_names,
    start=1
):

    print("=" * 60)
    print(
        f"[{index}/33] {tracking_name}"
    )
    print("=" * 60)


    # --------------------------------------
    # _tracking.csv を取り除く
    # --------------------------------------

    if tracking_name.endswith(
        "_tracking.csv"
    ):

        video_name = (
            tracking_name[
                :-len("_tracking.csv")
            ]
            + ".mp4"
        )

    else:

        video_name = tracking_name


    print(
        "元動画:",
        video_name
    )


    # --------------------------------------
    # 元動画検索
    # --------------------------------------

    video_matches = list(
        VIDEO_DIR.rglob(
            video_name
        )
    )


    if not video_matches:

        print(
            "元動画が見つかりません"
        )

        continue


    video_path = video_matches[0]


    # --------------------------------------
    # 動画オープン
    # --------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        print(
            "動画を開けません"
        )

        continue


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 29.97


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    rows = []

    frame_no = 0
    detected_frames = 0


    # ======================================
    # フレームごとに検出
    # ======================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        # ----------------------------------
        # train-11で検出
        # ----------------------------------

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )


        result = results[0]


        # ----------------------------------
        # 検出あり
        # ----------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            confidence_array = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            # confidence最大の検出を使用

            best_index = (
                confidence_array.argmax()
            )


            confidence = float(
                confidence_array[
                    best_index
                ]
            )


            xyxy = (
                result.boxes.xyxy[
                    best_index
                ]
                .cpu()
                .numpy()
            )


            x1, y1, x2, y2 = xyxy


            center_x = (
                x1 + x2
            ) / 2


            center_y = (
                y1 + y2
            ) / 2


            box_width = (
                x2 - x1
            )


            box_height = (
                y2 - y1
            )


            rows.append(
                {
                    "frame": frame_no,
                    "time_sec": (
                        frame_no / fps
                    ),
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": box_width,
                    "height": box_height,
                    "confidence": confidence,
                }
            )


            detected_frames += 1


        frame_no += 1


        # ----------------------------------
        # 進捗
        # ----------------------------------

        if frame_no % 50 == 0:

            print(
                f"{frame_no}/{total_frames} "
                f"| 検出 {detected_frames}"
            )


    cap.release()


    # ======================================
    # CSV保存
    # ======================================

    tracking_df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "time_sec",
            "center_x",
            "center_y",
            "width",
            "height",
            "confidence",
        ]
    )


    output_path = (
        OUTPUT_DIR
        / f"{Path(video_name).stem}_tracking.csv"
    )


    tracking_df.to_csv(
        output_path,
        index=False
    )


    # ======================================
    # 検出率
    # ======================================

    detection_rate = (

        detected_frames / frame_no

        if frame_no > 0

        else 0

    )


    print()
    print("完了")
    print(
        "全フレーム:",
        frame_no
    )
    print(
        "検出フレーム:",
        detected_frames
    )
    print(
        "検出率:",
        f"{detection_rate * 100:.1f}%"
    )
    print(
        "保存:",
        output_path
    )


    summary.append(
        {
            "video": video_name,
            "frames": frame_no,
            "detected_frames": detected_frames,
            "detection_rate": detection_rate,
        }
    )


# ==========================================
# サマリー
# ==========================================

summary_df = pd.DataFrame(
    summary
)


summary_path = (
    OUTPUT_DIR
    / "train11_tracking_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


print()
print("=" * 60)
print(
    "train-11 33動画の再追跡完了"
)
print("=" * 60)

print(
    "処理完了動画数:",
    len(summary_df)
)

print(
    "保存先:",
    OUTPUT_DIR
)

print(
    "サマリー:",
    summary_path
)