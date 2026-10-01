import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# 投球速度解析 v6
#
# v5のボール追跡CSVを使用
#
# ・隣接フレームの差分速度だけに頼らない
# ・軌跡全体を平滑化
# ・x(t), y(t) を多項式でフィッティング
# ・フィッティングした軌跡を時間微分
# ・px/sで速度を算出
#
# ※この段階ではまだkm/hへ変換しない
# ============================================================


# ============================================================
# 設定
# ============================================================

INPUT_DIR = Path("output/ball_tracking_v5")

OUTPUT_DIR = Path("output/pitch_speed_v6")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# フィッティング次数
#
# 2 = 放物線
# 3 = 3次曲線
#
# 今回は3次を使用
POLY_DEGREE = 3


# グラフを表示するか
SHOW_GRAPH = True


# ============================================================
# CSVを読み込む
# ============================================================


def load_tracking_csv(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = ["frame", "time_sec", "cx", "cy", "confidence", "interpolated"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:

        print()
        print("必要な列がありません:")
        print(missing)

        return None

    return df


# ============================================================
# 軌跡フィッティング
# ============================================================


def fit_trajectory(df):

    t = df["time_sec"].values.astype(float)

    x = df["cx"].values.astype(float)

    y = df["cy"].values.astype(float)

    # 時間を0スタートにする
    t0 = t[0]

    t_relative = t - t0

    # --------------------------------------------------------
    # x(t)
    # --------------------------------------------------------

    coeff_x = np.polyfit(t_relative, x, POLY_DEGREE)

    poly_x = np.poly1d(coeff_x)

    # --------------------------------------------------------
    # y(t)
    # --------------------------------------------------------

    coeff_y = np.polyfit(t_relative, y, POLY_DEGREE)

    poly_y = np.poly1d(coeff_y)

    # --------------------------------------------------------
    # 微分
    #
    # dx/dt
    # dy/dt
    # --------------------------------------------------------

    velocity_x = np.polyder(poly_x)

    velocity_y = np.polyder(poly_y)

    return (t_relative, poly_x, poly_y, velocity_x, velocity_y)


# ============================================================
# 速度計算
# ============================================================


def calculate_speed(t_relative, velocity_x, velocity_y):

    vx = velocity_x(t_relative)

    vy = velocity_y(t_relative)

    speed = np.sqrt(vx**2 + vy**2)

    return vx, vy, speed


# ============================================================
# グラフ作成
# ============================================================


def create_graph(df, t_relative, poly_x, poly_y, speed, output_path, title):

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ========================================================
    # 上：ボール軌跡
    # ========================================================

    ax = axes[0]

    # 実際の検出点
    real = df[df["interpolated"] == 0]

    # 補間点
    interpolated = df[df["interpolated"] == 1]

    ax.scatter(real["cx"], real["cy"], label="Actual detection", s=30)

    if len(interpolated) > 0:

        ax.scatter(
            interpolated["cx"],
            interpolated["cy"],
            label="Interpolated",
            s=40,
            marker="x",
        )

    # 滑らかな軌跡
    t_plot = np.linspace(t_relative.min(), t_relative.max(), 300)

    x_plot = poly_x(t_plot)

    y_plot = poly_y(t_plot)

    ax.plot(x_plot, y_plot, label="Fitted trajectory")

    ax.invert_yaxis()

    ax.set_xlabel("X [pixel]")

    ax.set_ylabel("Y [pixel]")

    ax.set_title(title + " - Trajectory")

    ax.legend()

    ax.grid(True, alpha=0.3)

    # ========================================================
    # 下：速度
    # ========================================================

    ax2 = axes[1]

    ax2.plot(t_relative, speed)

    ax2.set_xlabel("Time from tracking start [s]")

    ax2.set_ylabel("Speed [px/s]")

    ax2.set_title("Trajectory-derived speed")

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_path, dpi=200)

    if SHOW_GRAPH:

        plt.show()

    plt.close()


# ============================================================
# 1動画解析
# ============================================================


def analyze_video(csv_path):

    print()
    print("============================================================")
    print("動画:", csv_path.stem)
    print("============================================================")

    df = load_tracking_csv(csv_path)

    if df is None:

        return None

    if len(df) < 7:

        print("追跡点が少なすぎます")

        return None

    print(f"追跡点数: {len(df)}")

    # ========================================================
    # 軌跡フィッティング
    # ========================================================

    t_relative, poly_x, poly_y, velocity_x, velocity_y = fit_trajectory(df)

    # ========================================================
    # 速度
    # ========================================================

    vx, vy, speed = calculate_speed(t_relative, velocity_x, velocity_y)

    # ========================================================
    # 結果DataFrame
    # ========================================================

    result = df.copy()

    result["vx_px_s"] = vx

    result["vy_px_s"] = vy

    result["speed_px_s"] = speed

    # ========================================================
    # 平均・最大
    # ========================================================

    average_speed = np.mean(speed)

    maximum_speed = np.max(speed)

    max_index = np.argmax(speed)

    max_frame = int(result.iloc[max_index]["frame"])

    # ========================================================
    # フィッティング軌跡の始点・終点
    # ========================================================

    start_x = poly_x(t_relative[0])

    start_y = poly_y(t_relative[0])

    end_x = poly_x(t_relative[-1])

    end_y = poly_y(t_relative[-1])

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # ========================================================
    # 表示
    # ========================================================

    print()
    print("------------------------------------------")
    print("軌跡フィッティング")
    print("------------------------------------------")

    print(f"フィッティング次数: " f"{POLY_DEGREE}")

    print(f"追跡時間: " f"{t_relative[-1]:.4f} sec")

    print(f"軌跡移動距離: " f"{distance:.2f} px")

    print()
    print("------------------------------------------")
    print("速度解析")
    print("------------------------------------------")

    print(f"平均速度      : " f"{average_speed:.2f} px/s")

    print(f"最大速度      : " f"{maximum_speed:.2f} px/s")

    print(f"最大速度frame : " f"{max_frame}")

    # ========================================================
    # CSV保存
    # ========================================================

    output_csv = OUTPUT_DIR / f"{csv_path.stem}_v6.csv"

    result.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # ========================================================
    # グラフ
    # ========================================================

    output_png = OUTPUT_DIR / f"{csv_path.stem}_v6.png"

    create_graph(df, t_relative, poly_x, poly_y, speed, output_png, csv_path.stem)

    print()
    print("CSV:", output_csv)

    print("グラフ:", output_png)

    return {
        "name": csv_path.stem,
        "points": len(df),
        "average": average_speed,
        "maximum": maximum_speed,
        "max_frame": max_frame,
        "distance": distance,
    }


# ============================================================
# メイン
# ============================================================

print()
print("============================================================")
print("ボール追跡 v6")
print("軌跡全体フィッティングによる速度解析")
print("============================================================")

print()
print("入力:")
print(INPUT_DIR)

print()
print("出力:")
print(OUTPUT_DIR)

print()
print(f"フィッティング次数: " f"{POLY_DEGREE}")


csv_files = sorted(INPUT_DIR.glob("*_tracking.csv"))


if not csv_files:

    print()
    print("追跡CSVが見つかりません")

    print("確認:", INPUT_DIR)

    raise SystemExit


results = []


for csv_path in csv_files:

    result = analyze_video(csv_path)

    if result is not None:

        results.append(result)


# ============================================================
# 最終結果
# ============================================================

print()
print("============================================================")
print("全動画の速度解析完了")
print("============================================================")

print()

for result in results:

    print(
        f"{result['name']} | "
        f"平均: {result['average']:.2f} px/s | "
        f"最大: {result['maximum']:.2f} px/s | "
        f"最大Frame: {result['max_frame']}"
    )

print()
print("出力先:", OUTPUT_DIR)
