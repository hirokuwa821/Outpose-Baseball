import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 設定
# ==========================================

CSV_FILES = [
    "output/ball_tracking_v2/pitch_102_1_tracking.csv",
    "output/ball_tracking_v2/pitch_98_1_tracking.csv",
    "output/ball_tracking_v2/pitch_98_2_tracking.csv",
]

OUTPUT_DIR = Path("output/ball_trajectory")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 各CSVを解析
# ==========================================

for csv_path in CSV_FILES:

    path = Path(csv_path)

    if not path.exists():
        print()
        print("CSVがありません:")
        print(csv_path)
        continue

    print()
    print("=" * 60)
    print("解析:", csv_path)
    print("=" * 60)

    df = pd.read_csv(path)

    if df.empty:
        print("データがありません")
        continue

    # --------------------------------------
    # データ確認
    # --------------------------------------

    print("追跡点数:", len(df))

    print(
        "Frame:",
        int(df["frame"].iloc[0]),
        "→",
        int(df["frame"].iloc[-1]),
    )

    print("Confidence平均:", f"{df['confidence'].mean():.3f}")

    # ======================================
    # フレーム間移動距離
    # ======================================

    dx = df["center_x"].diff()
    dy = df["center_y"].diff()

    df["move_distance"] = (dx**2 + dy**2) ** 0.5

    # 最初のフレームは比較できない
    df.loc[df.index[0], "move_distance"] = 0

    print("総移動距離:", f"{df['move_distance'].sum():.2f} px")

    print("平均フレーム間移動:", f"{df['move_distance'].iloc[1:].mean():.2f} px")

    # ======================================
    # 軌跡グラフ
    # ======================================

    plt.figure(figsize=(8, 10))

    plt.plot(
        df["center_x"],
        df["center_y"],
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    # 開始点
    plt.scatter(
        df["center_x"].iloc[0],
        df["center_y"].iloc[0],
        s=100,
        marker="o",
        label="Start",
    )

    # 終了点
    plt.scatter(
        df["center_x"].iloc[-1],
        df["center_y"].iloc[-1],
        s=100,
        marker="x",
        label="End",
    )

    # フレーム番号を表示
    for _, row in df.iterrows():

        plt.text(
            row["center_x"],
            row["center_y"],
            str(int(row["frame"])),
            fontsize=8,
        )

    plt.xlabel("X position (pixel)")
    plt.ylabel("Y position (pixel)")

    plt.title(f"Ball trajectory - {path.stem}")

    # 画像座標なのでY軸を反転
    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{path.stem}_trajectory.png"

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print()
    print("軌跡グラフ:")
    print(output_path)

    # ======================================
    # フレーム間移動距離グラフ
    # ======================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["frame"],
        df["move_distance"],
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    plt.xlabel("Frame")
    plt.ylabel("Movement (pixel/frame)")

    plt.title(f"Ball movement - {path.stem}")

    plt.grid(True)

    plt.tight_layout()

    output_path2 = OUTPUT_DIR / f"{path.stem}_movement.png"

    plt.savefig(
        output_path2,
        dpi=200,
    )

    plt.close()

    print("移動量グラフ:")
    print(output_path2)


print()
print("=" * 60)
print("解析完了")
print("=" * 60)

print()
print("保存先:")
print("output/ball_trajectory")
