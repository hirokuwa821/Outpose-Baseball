# ------------------------------------------------------------
# src/gui_app.py – Simple Tkinter GUI for pitch speed analysis
# ------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading

# Import the analysis module (relative import works when run as package)
import sys, os
# Ensure src directory is in sys.path when running as a script
sys.path.append(os.path.dirname(__file__))
import pitch_speed_v23_auto_distance as ps


def run_analysis(video_name: str, is_left: bool, auto_detect: bool, unit: str = "kmh") -> str:
    """Execute analysis for a single video and return a formatted string.
    Supports unit selection (km/h or mph) and provides a simple OpenCV fallback
    for ball‑tracking CSV if the file is missing.
    """
    if video_name not in ps.BASE_FRAMES:
        return f"[Error] 動画 '{video_name}' が定義されていません。"

    # Update handedness for this run only
    ps.HANDEDNESS[video_name] = is_left

    # Load keypoints
    if not ps.KEYPOINTS_PATH.exists():
        return f"[Error] キーポイントファイルが見つかりません: {ps.KEYPOINTS_PATH}"
    kp_df = ps.pd.read_csv(ps.KEYPOINTS_PATH)
    y_net = ps.detect_net_y(kp_df)

    release_frame, net_frame = ps.BASE_FRAMES[video_name]
    # 足座標取得（auto_detect フラグを渡す）
    y_plate, y_foot = ps.get_foot_coordinates(kp_df, release_frame, video_name, is_left, auto_detect)
    if y_plate is None:
        return "[Error] 足座標取得に失敗しました。"

    pitch_distance, extension = ps.calculate_dynamic_distance(
        y_plate, y_net, y_foot, ps.TOTAL_FIELD_DISTANCE_M
    )

    # ボールトラッキング CSV 読み込み – if missing, try a lightweight OpenCV detection
    ball_csv = ps.INPUT_DIR / f"{video_name}_tracking.csv"
    if ball_csv.exists():
        rows = ps.load_ball_csv(ball_csv)
    else:
        try:
            import cv2
            video_path = ps.INPUT_DIR.parent / f"{video_name}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            rows = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), r) = cv2.minEnclosingCircle(c)
                    if r > 2:
                        rows.append({"frame": frame_idx, "time_sec": frame_idx / fps})
                frame_idx += 1
            cap.release()
        except Exception as e:
            return f"[Error] ボールトラッキングに失敗: {e}"
    if len(rows) < 2:
        return f"[Warning] ボールトラッキングデータが不足しています: {video_name}"

    frames = ps.np.array([r["frame"] for r in rows])
    times = ps.np.array([r["time_sec"] for r in rows])
    time_model = ps.np.poly1d(ps.np.polyfit(frames, times, 1))
    flight_time = time_model(net_frame) - time_model(release_frame)
    if flight_time <= 0:
        return f"[Warning] 飛行時間が負です: {video_name}"

    release_speed_mps = (
        (ps.math.exp(ps.K_DRAG * pitch_distance) - 1)
        / (ps.K_DRAG * flight_time)
    )
    release_speed_kmh = release_speed_mps * 3.6
    release_speed_mph = release_speed_kmh * 0.621371

    # Choose display unit
    if unit == "mph":
        speed_val = release_speed_mph
        speed_label = "mph"
    else:
        speed_val = release_speed_kmh
        speed_label = "km/h"

    hand_str = "左投" if is_left else "右投"
    result = (
        f"動画: {video_name}\n"
        f"投手: {hand_str}\n"
        f"AI取得座標: プレートY={y_plate:.1f}, 踏み出し足Y={y_foot:.1f}\n"
        f"エクステンション: {extension:.2f} m   投球距離: {pitch_distance:.2f} m\n"
        f"飛行時間: {flight_time:.4f} s   初速推定: {speed_val:.2f} {speed_label}\n"
    )
    return result


def start_analysis_thread(video_var, hand_var, auto_var, unit_var, output_widget):
    def task():
        output_widget.delete('1.0', tk.END)
        output_widget.insert(tk.END, "解析中…\n")
        try:
            res = run_analysis(video_var.get(), hand_var.get(), auto_var.get(), unit=unit_var.get())
            output_widget.delete('1.0', tk.END)
            output_widget.insert(tk.END, res)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    threading.Thread(target=task, daemon=True).start()


def main():
    root = tk.Tk()
    root.title("投球速度解析 GUI")
    root.geometry("600x500")

    # Video selection combobox
    ttk.Label(root, text="動画選択:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    video_var = tk.StringVar(value=list(ps.BASE_FRAMES.keys())[0])
    video_cb = ttk.Combobox(root, textvariable=video_var, values=list(ps.BASE_FRAMES.keys()), state="readonly")
    video_cb.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    # Handedness checkbox
    ttk.Label(root, text="投手の左右:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    hand_var = tk.BooleanVar(value=False)  # False = 右投, True = 左投
    ttk.Checkbutton(root, text="左投げ", variable=hand_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)

    # Auto‑detect flag
    auto_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(root, text="自動 PLATE フレーム探索", variable=auto_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    # Speed unit selection (km/h or mph)
    ttk.Label(root, text="速度単位:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    unit_var = tk.StringVar(value="kmh")
    ttk.Radiobutton(root, text="km/h", variable=unit_var, value="kmh").grid(row=3, column=1, sticky="w", padx=5)
    ttk.Radiobutton(root, text="mph", variable=unit_var, value="mph").grid(row=3, column=1, sticky="e", padx=5)

    # Run button
    run_btn = ttk.Button(
        root,
        text="解析開始",
        command=lambda: start_analysis_thread(video_var, hand_var, auto_var, unit_var, output_text),
    )
    run_btn.grid(row=4, column=0, columnspan=2, pady=10)

    # Output area
    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20)
    output_text.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    # Make UI resize nicely
    root.grid_rowconfigure(5, weight=1)
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()
