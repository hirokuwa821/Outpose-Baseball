import cv2
from pathlib import Path
import pandas as pd

# ============================================================
# 投球速度解析 v10
#
# リリースフレームとホーム到達フレームを手動指定
# ------------------------------------------------------------
# 目的：
# 自動追跡区間をそのまま球速計算に使わず、
# 実際の投球区間を人間が指定して飛行時間を求める。
#
# R : リリースフレームを現在位置に設定
# H : ホーム到達フレームを現在位置に設定
# S : 保存して次の動画へ
# N : 1フレーム進む
# B : 1フレーム戻る
# Q : 終了
#
# マウスクリック：
# ・左クリック → 現在フレームをリリースに設定
# ・右クリック → 現在フレームをホーム到達に設定
# ============================================================


VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("output/pitch_speed_v10")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 設定
# ============================================================

PITCHING_DISTANCE_M = 18.44

VIDEO_EXTENSIONS = ["*.mp4", "*.MP4", "*.avi", "*.mov"]


# ============================================================
# 動画一覧
# ============================================================

videos = []

for ext in VIDEO_EXTENSIONS:
    videos.extend(VIDEO_DIR.glob(ext))

videos = sorted(videos)


if not videos:

    print("動画が見つかりません")

    raise SystemExit


# ============================================================
# 動画ごとの解析
# ============================================================

results = []


for video_path in videos:

    print()
    print("=" * 60)
    print("動画:", video_path.name)
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません:", video_path)

        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps:.4f}")

    print(f"サイズ: {width} x {height}")

    print(f"総フレーム数: {total_frames}")

    # --------------------------------------------------------
    # 状態
    # --------------------------------------------------------

    current_frame = 0

    release_frame = None
    contact_frame = None

    window_name = "Pitch Speed v10"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ========================================================
    # マウス
    # ========================================================

    def mouse_callback(event, x, y, flags, param):

        global release_frame
        global contact_frame

        if event == cv2.EVENT_LBUTTONDOWN:

            release_frame = current_frame

            print()
            print(f"リリース設定: " f"Frame {release_frame}")

        elif event == cv2.EVENT_RBUTTONDOWN:

            contact_frame = current_frame

            print()
            print(f"ホーム到達設定: " f"Frame {contact_frame}")

    cv2.setMouseCallback(window_name, mouse_callback)

    # ========================================================
    # フレーム表示
    # ========================================================

    while True:

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()

        if not ret:

            break

        display = frame.copy()

        # ----------------------------------------------------
        # リリースフレーム表示
        # ----------------------------------------------------

        if release_frame is not None:

            cv2.putText(
                display,
                f"Release: {release_frame}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # ----------------------------------------------------
        # ホーム到達表示
        # ----------------------------------------------------

        if contact_frame is not None:

            cv2.putText(
                display,
                f"Home: {contact_frame}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ----------------------------------------------------
        # 現在フレーム
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Frame: {current_frame}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # 操作説明
        # ----------------------------------------------------

        cv2.putText(
            display,
            "LeftClick=Release  RightClick=Home",
            (10, height - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "N=Next  B=Back  S=Save  Q=Quit",
            (10, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        # ====================================================
        # 次フレーム
        # ====================================================

        if key == ord("n"):

            current_frame = min(current_frame + 1, total_frames - 1)

        # ====================================================
        # 前フレーム
        # ====================================================

        elif key == ord("b"):

            current_frame = max(current_frame - 1, 0)

        # ====================================================
        # R
        # 現在フレームをリリースに設定
        # ====================================================

        elif key == ord("r"):

            release_frame = current_frame

            print("リリース:", release_frame)

        # ====================================================
        # H
        # 現在フレームをホーム到達に設定
        # ====================================================

        elif key == ord("h"):

            contact_frame = current_frame

            print("ホーム到達:", contact_frame)

        # ====================================================
        # S
        # 保存
        # ====================================================

        elif key == ord("s"):

            if release_frame is None or contact_frame is None:

                print()
                print("リリースとホーム到達の" "両方を設定してください")

                continue

            if contact_frame <= release_frame:

                print()
                print("エラー:")

                print("ホーム到達フレームは" "リリースより後にしてください")

                continue

            # ------------------------------------------------
            # 飛行時間
            # ------------------------------------------------

            frame_difference = contact_frame - release_frame

            flight_time = frame_difference / fps

            # ------------------------------------------------
            # 平均速度
            # ------------------------------------------------

            speed_m_s = PITCHING_DISTANCE_M / flight_time

            speed_km_h = speed_m_s * 3.6

            # ------------------------------------------------
            # 結果表示
            # ------------------------------------------------

            print()
            print("------------------------------------------")

            print("投球速度解析")

            print("------------------------------------------")

            print(f"リリースFrame : " f"{release_frame}")

            print(f"ホームFrame   : " f"{contact_frame}")

            print(f"フレーム差    : " f"{frame_difference}")

            print(f"FPS           : " f"{fps:.4f}")

            print(f"飛行時間      : " f"{flight_time:.5f} sec")

            print(f"距離          : " f"{PITCHING_DISTANCE_M:.2f} m")

            print(f"平均速度      : " f"{speed_m_s:.3f} m/s")

            print(f"平均速度      : " f"{speed_km_h:.2f} km/h")

            print("※18.44 mを投球距離として仮定")

            print("------------------------------------------")

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            result = {
                "video": video_path.name,
                "fps": fps,
                "release_frame": release_frame,
                "contact_frame": contact_frame,
                "frame_difference": frame_difference,
                "flight_time_sec": flight_time,
                "distance_m": PITCHING_DISTANCE_M,
                "speed_m_s": speed_m_s,
                "speed_km_h": speed_km_h,
            }

            results.append(result)

            result_df = pd.DataFrame([result])

            output_csv = OUTPUT_DIR / f"{video_path.stem}_v10.csv"

            result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

            print()
            print("CSV:", output_csv)

            break

        # ====================================================
        # Q
        # ====================================================

        elif key == ord("q"):

            cap.release()

            cv2.destroyAllWindows()

            print()
            print("終了しました")

            raise SystemExit

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# 全結果
# ============================================================

print()
print("=" * 60)
print("全動画の投球速度解析完了")
print("=" * 60)


if results:

    for result in results:

        print()

        print(f"{result['video']}")

        print(f"Release: " f"{result['release_frame']}")

        print(f"Home: " f"{result['contact_frame']}")

        print(f"飛行時間: " f"{result['flight_time_sec']:.5f} sec")

        print(f"速度: " f"{result['speed_km_h']:.2f} km/h")

    # --------------------------------------------------------
    # 全結果CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(results)

    summary_csv = OUTPUT_DIR / "pitch_speed_v10_summary.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print()
    print("まとめCSV:", summary_csv)


else:

    print("保存された結果はありません")


print()
print("出力先:", OUTPUT_DIR)
