import pandas as pd
import matplotlib.pyplot as plt


def visualize(release_frame):

    df = pd.read_csv("output/features.csv")

    plt.figure(figsize=(12,6))

    plt.plot(df["frame"], df["wrist_speed"], label="Wrist Speed")
    plt.plot(df["frame"], df["elbow_speed"], label="Elbow Speed")
    plt.plot(df["frame"], df["shoulder_speed"], label="Shoulder Speed")

    plt.axvline(
        release_frame,
        color="red",
        linestyle="--",
        label="Release"
    )

    plt.xlabel("Frame")
    plt.ylabel("Speed (pixel/frame)")
    plt.title("Pitch Analysis")

    plt.legend()

    plt.grid(True)

    plt.savefig("output/speed_graph.png")

    plt.close()

    print("グラフを保存しました！")