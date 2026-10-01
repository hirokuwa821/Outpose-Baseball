# ------------------------------------------------------------
# src/app_cli.py – Command Line Interface for pitch speed analysis
# ------------------------------------------------------------
import argparse
import importlib.util
import sys
from pathlib import Path

# Dynamically load the analysis module
SCRIPT_PATH = Path(__file__).parent / "pitch_speed_v23_auto_distance.py"
spec = importlib.util.spec_from_file_location("pitch_speed_v23_auto_distance", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="投球速度解析 CLI")
    parser.add_argument("-v", "--video", required=True, help="動画名 (例: pitch_102_1)")
    parser.add_argument("-l", "--left-handed", action="store_true", help="左投げの場合に指定 (デフォルトは右投げ)")
    parser.add_argument("--auto-detect", action="store_true", help="支え足フレームを自動探索する")
    parser.add_argument("--unit", choices=["kmh", "mph"], default="kmh", help="速度単位: km/h または mph (デフォルト kmh)")
    return parser.parse_args()

def main():
    args = parse_args()
    video = args.video
    if video not in module.BASE_FRAMES:
        # Allow ad‑hoc videos captured via the webcam. Use the first and last
        # frames of the keypoints dataframe as a fallback.
        release_frame = 0
        net_frame = len(kp_df) - 1
        print(f"[Warning] 動画 '{video}' は定義されていません。自動的にフレーム範囲 (0-{net_frame}) を使用します。")
    else:
        release_frame, net_frame = module.BASE_FRAMES[video]

    # Load keypoints CSV
    if not module.KEYPOINTS_PATH.exists():
        sys.exit(f"[Error] キーポイントファイルが見つかりません: {module.KEYPOINTS_PATH}")
    kp_df = module.pd.read_csv(module.KEYPOINTS_PATH)
    y_net = module.detect_net_y(kp_df)
    release_frame, net_frame = module.BASE_FRAMES[video]
    # Get foot coordinates (auto-detect flag passed)
    y_plate, y_foot = module.get_foot_coordinates(
        kp_df, release_frame, video, args.left_handed, auto_detect=args.auto_detect
    )
    if y_plate is None:
        sys.exit("[Error] 足座標取得に失敗しました")
    pitch_distance, extension = module.calculate_dynamic_distance(
        y_plate, y_net, y_foot, module.TOTAL_FIELD_DISTANCE_M
    )
    # Load ball tracking CSV
    ball_csv = module.INPUT_DIR / f"{video}_tracking.csv"
    rows = module.load_ball_csv(ball_csv)
    if len(rows) < 2:
        sys.exit(f"[Warning] ボールトラッキングデータが不足しています: {video}")
    frames = module.np.array([r["frame"] for r in rows])
    times = module.np.array([r["time_sec"] for r in rows])
    time_model = module.np.poly1d(module.np.polyfit(frames, times, 1))
    flight_time = time_model(net_frame) - time_model(release_frame)
    if flight_time <= 0:
        sys.exit(f"[Warning] 飛行時間が負です: {video}")
    release_speed_mps = (
        (module.math.exp(module.K_DRAG * pitch_distance) - 1)
        / (module.K_DRAG * flight_time)
    )
import result_logger
# ... after calculating release_speed_kmh
    # Log the result for history tracking
    result_logger.log_result(
        video,
        args.left_handed,
        release_speed_kmh,
        extension,
        pitch_distance,
        flight_time,
    )
    print(f"動画: {video}")
    print(f"投手: {hand_str}")
    print(f"AI取得座標: プレートY={y_plate:.1f}, 踏み出し足Y={y_foot:.1f}")
    print(f"エクステンション: {extension:.2f} m   投球距離: {pitch_distance:.2f} m")
    print(f"飛行時間: {flight_time:.4f} s   初速推定: {release_speed_kmh:.2f} km/h")
    print("=" * 60)

if __name__ == "__main__":
    main()
