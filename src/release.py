import pandas as pd
import numpy as np


def detect_release(hand="right", csv_path="output/keypoints.csv"):

    df = pd.read_csv(csv_path)

    speed = []

    if hand == "right":
        wrist_x = "x_10"
        wrist_y = "y_10"
    else:
        wrist_x = "x_9"
        wrist_y = "y_9"

    for i in range(1, len(df)):

        dx = df.loc[i, wrist_x] - df.loc[i - 1, wrist_x]
        dy = df.loc[i, wrist_y] - df.loc[i - 1, wrist_y]

        v = np.sqrt(dx**2 + dy**2)

        speed.append(v)

    release_frame = np.argmax(speed) + 1

    return release_frame