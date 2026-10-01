import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = Path("output/ball_tracking_v2")
OUTPUT_DIR = Path("output/ball_trajectory_plots")

OUTPUT_DIR.mkdir(exist_ok=True)

# 予測誤差が大きかった動画
TARGETS = [
    "mlb_007_84.1mph_0JMTIRJHGH8V_right_tracking.csv",
    "mlb_006_81.9mph_VKUS2DQWEUCN_right_tracking.csv",
    "mlb_109_99.6mph_K6WDTHLHGK51_tracking.csv",
    "mlb_122_101.4mph_GD2COQK31DXN_tracking.csv",
    "mlb_001_81.1mph_CGI1SSOSP466_right_tracking.csv",
    "mlb_508_95.6mph_XRRC5RDOCT74_tracking.csv",
    "mlb_019_89.5mph_X81T4GSP3IKJ_left_tracking.csv",
    "mlb_070_86.5mph_I94YTJBIRFFU_tracking.csv",
]

for filename in TARGETS:
    path = INPUT_DIR / filename

    if not path.exists():
        print("見つからない:", filename)
        continue

    df = pd.read_csv(path)

    plt.figure(figsize=(7, 5))
    plt.plot(df["center_x"], df["center_y"], marker="o")

    plt.gca().invert_yaxis()

    plt.xlabel("X position (pixel)")
    plt.ylabel("Y position (pixel)")
    plt.title(filename.replace("_tracking.csv", ""))

    plt.grid(True)

    output_path = OUTPUT_DIR / filename.replace("_tracking.csv", ".png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

print()
print("軌跡グラフを保存しました:")
print(OUTPUT_DIR)
