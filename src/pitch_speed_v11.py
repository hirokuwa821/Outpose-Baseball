import cv2
from pathlib import Path
import pandas as pd

# ============================================================
# 投球速度解析 v11
#
# リリース位置を考慮した投球距離で速度を計算
#
# 18.44 m
#   ↓
# ピッチャーズプレート～ホームプレート間距離
#
# そこからリリース位置までの前方距離を引く。
#
# 例：
# 18.44 m - 0.60 m = 17.84 m
#
# ※リリース位置の前方距離は手動入力
# ※カメラの透視歪みはまだ補正していない
# ============================================================


VIDEO_DIR = Path("videos")

OUTPUT_DIR = Path("output/pitch_speed_v11")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 基本設定
# ============================================================

PLATE_DISTANCE_M = 18.44

# ------------------------------------------------------------
# リリース位置
#
# ピッチャーズプレート前縁から、
# ホーム方向へ何m前でリリースしたか。
#
# 分からない場合はまず 0.0 にする。
# ------------------------------------------------------------

RELEASE_FORWARD_M = 1.95


# ============================================================
# 動画一覧
# ============================================================

videos = sorted(VIDEO_DIR.glob("*.mp4"))


if not videos:

    print("videosフォルダにmp4がありません")

    raise SystemExit


# ============================================================
# 投球距離
# ============================================================

PITCH_DISTANCE_M = PLATE_DISTANCE_M - RELEASE_FORWARD_M


print()
print("=" * 60)
print("投球速度解析 v11")
print("リリース位置補正")
print("=" * 60)

print()
print(f"ピッチャーズプレート～ホーム: " f"{PLATE_DISTANCE_M:.2f} m")

print(f"リリース前方距離: " f"{RELEASE_FORWARD_M:.2f} m")

print(f"使用投球距離: " f"{PITCH_DISTANCE_M:.2f} m")

print()
print("※カメラの透視歪みは未補正")

print("※リリース前方距離は仮定値")


# ============================================================
# 結果
# ============================================================

results = []


# ============================================================
# 動画ごと
# ============================================================

for video_path in videos:

    print()
    print("=" * 60)
    print("動画:", video_path.name)
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("動画を開けません")

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

    home_frame = None

    window_name = "Pitch Speed v11"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ========================================================
    # マウス
    # ========================================================

    def mouse_callback(event, x, y, flags, param):

        global release_frame
        global home_frame

        if event == cv2.EVENT_LBUTTONDOWN:

            release_frame = current_frame

            print(f"リリース設定: " f"Frame {release_frame}")

        elif event == cv2.EVENT_RBUTTONDOWN:

            home_frame = current_frame

            print(f"ホーム到達設定: " f"Frame {home_frame}")

    cv2.setMouseCallback(window_name, mouse_callback)

    # ========================================================
    # フレーム操作
    # ========================================================

    while True:

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()

        if not ret:

            break

        display = frame.copy()

        # ----------------------------------------------------
        # 現在フレーム
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Frame: {current_frame}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # リリース
        # ----------------------------------------------------

        if release_frame is not None:

            cv2.putText(
                display,
                f"Release: {release_frame}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # ----------------------------------------------------
        # ホーム
        # ----------------------------------------------------

        if home_frame is not None:

            cv2.putText(
                display,
                f"Home: {home_frame}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ----------------------------------------------------
        # 操作
        # ----------------------------------------------------

        cv2.putText(
            display,
            "LeftClick=Release",
            (10, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "RightClick=Home",
            (10, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display,
            "N=Next B=Back S=Save Q=Quit",
            (10, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # ----------------------------------------------------
        # 次
        # ----------------------------------------------------

        if key == ord("n"):

            current_frame = min(current_frame + 1, total_frames - 1)

        # ----------------------------------------------------
        # 前
        # ----------------------------------------------------

        elif key == ord("b"):

            current_frame = max(current_frame - 1, 0)

        # ----------------------------------------------------
        # R
        # ----------------------------------------------------

        elif key == ord("r"):

            release_frame = current_frame

            print(f"リリース: " f"{release_frame}")

        # ----------------------------------------------------
        # H
        # ----------------------------------------------------

        elif key == ord("h"):

            home_frame = current_frame

            print(f"ホーム: " f"{home_frame}")

        # ----------------------------------------------------
        # S
        # ----------------------------------------------------

        elif key == ord("s"):

            if release_frame is None or home_frame is None:

                print()
                print("リリースとホーム到達を" "両方指定してください")

                continue

            if home_frame <= release_frame:

                print()
                print("ホーム到達フレームは" "リリースより後にしてください")

                continue

            # =================================================
            # 飛行時間
            # =================================================

            frame_difference = home_frame - release_frame

            flight_time = frame_difference / fps

            # =================================================
            # 速度
            # =================================================

            speed_m_s = PITCH_DISTANCE_M / flight_time

            speed_km_h = speed_m_s * 3.6

            # =================================================
            # 結果
            # =================================================

            print()
            print("------------------------------------------")

            print("投球速度解析 v11")

            print("------------------------------------------")

            print(f"リリースFrame : " f"{release_frame}")

            print(f"ホームFrame   : " f"{home_frame}")

            print(f"フレーム差    : " f"{frame_difference}")

            print(f"FPS           : " f"{fps:.4f}")

            print(f"飛行時間      : " f"{flight_time:.5f} sec")

            print(f"投球距離      : " f"{PITCH_DISTANCE_M:.3f} m")

            print(f"平均速度      : " f"{speed_m_s:.3f} m/s")

            print(f"平均速度      : " f"{speed_km_h:.2f} km/h")

            print("------------------------------------------")

            # =================================================
            # CSV
            # =================================================

            result = {
                "video": video_path.name,
                "fps": fps,
                "release_frame": release_frame,
                "home_frame": home_frame,
                "frame_difference": frame_difference,
                "flight_time_sec": flight_time,
                "plate_distance_m": PLATE_DISTANCE_M,
                "release_forward_m": RELEASE_FORWARD_M,
                "pitch_distance_m": PITCH_DISTANCE_M,
                "speed_m_s": speed_m_s,
                "speed_km_h": speed_km_h,
            }

            results.append(result)

            output_csv = OUTPUT_DIR / f"{video_path.stem}_v11.csv"

            pd.DataFrame([result]).to_csv(output_csv, index=False, encoding="utf-8-sig")

            print()
            print("CSV:", output_csv)

            break

        # ----------------------------------------------------
        # Q
        # ----------------------------------------------------

        elif key == ord("q"):

            cap.release()

            cv2.destroyAllWindows()

            print("終了しました")

            raise SystemExit

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# 全体結果
# ============================================================

print()
print("=" * 60)
print("v11 全動画解析完了")
print("=" * 60)


if results:

    for r in results:

        print()

        print(r["video"])

        print(f"Frame " f"{r['release_frame']} " f"→ " f"{r['home_frame']}")

        print(f"飛行時間: " f"{r['flight_time_sec']:.5f} sec")

        print(f"投球距離: " f"{r['pitch_distance_m']:.3f} m")

        print(f"速度: " f"{r['speed_km_h']:.2f} km/h")

    # --------------------------------------------------------
    # まとめCSV
    # --------------------------------------------------------

    summary_csv = OUTPUT_DIR / "pitch_speed_v11_summary.csv"

    pd.DataFrame(results).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print()
    print("まとめCSV:", summary_csv)


print()
print("出力先:", OUTPUT_DIR)
