from release import detect_release
import pandas as pd
import numpy as np
from pathlib import Path


def calc_angle(ax, ay, bx, by, cx, cy):

    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos = np.clip(cos, -1, 1)

    return np.degrees(np.arccos(cos))


def calc_speed(df, idx, point):

    if idx == 0:
        return 0

    dx = df.loc[idx, f"x_{point}"] - df.loc[idx-1, f"x_{point}"]
    dy = df.loc[idx, f"y_{point}"] - df.loc[idx-1, f"y_{point}"]

    return np.sqrt(dx**2 + dy**2)
def calc_acceleration(df, idx, point):

    if idx <= 1:
        return 0

    speed_now = calc_speed(df, idx, point)
    speed_prev = calc_speed(df, idx - 1, point)

    return speed_now - speed_prev
def calc_center_speed(df, idx):

    if idx == 0:
        return 0

    x1 = (df.loc[idx, "x_11"] + df.loc[idx, "x_12"]) / 2
    y1 = (df.loc[idx, "y_11"] + df.loc[idx, "y_12"]) / 2

    x0 = (df.loc[idx-1, "x_11"] + df.loc[idx-1, "x_12"]) / 2
    y0 = (df.loc[idx-1, "y_11"] + df.loc[idx-1, "y_12"]) / 2

    return np.sqrt((x1-x0)**2 + (y1-y0)**2)
def calc_distance(ax, ay, bx, by):

    return np.sqrt((ax - bx)**2 + (ay - by)**2)

def create_features(hand="right"):

    df = pd.read_csv("output/keypoints.csv")

    if hand == "right":

        shoulder = 6
        elbow = 8
        wrist = 10
        hip = 12
        knee = 14
        ankle = 16

    else:

        shoulder = 5
        elbow = 7
        wrist = 9
        hip = 11
        knee = 13
        ankle = 15
    max_wrist_speed = 0
    max_elbow_speed = 0
    max_shoulder_speed = 0
    max_hip_speed = 0
    max_knee_speed = 0
    max_ankle_speed = 0
    max_wrist_acc = 0
    max_elbow_acc = 0
    max_shoulder_acc = 0
    max_hip_acc = 0
    max_knee_acc = 0
    max_ankle_acc = 0
    rows = []
    print("初期値:", max_wrist_acc)

    for i, row in df.iterrows():

        elbow_angle = calc_angle(
               
            row[f"x_{shoulder}"], row[f"y_{shoulder}"],
            row[f"x_{elbow}"], row[f"y_{elbow}"],
            row[f"x_{wrist}"], row[f"y_{wrist}"]
        )
        shoulder_angle = calc_angle(
            row[f"x_{hip}"], row[f"y_{hip}"],
            row[f"x_{shoulder}"], row[f"y_{shoulder}"],
            row[f"x_{elbow}"], row[f"y_{elbow}"]
        )
        hip_angle = calc_angle(
            row[f"x_{shoulder}"], row[f"y_{shoulder}"],
            row[f"x_{hip}"], row[f"y_{hip}"],
            row[f"x_{knee}"], row[f"y_{knee}"]
        )
        knee_angle = calc_angle(
            row[f"x_{hip}"], row[f"y_{hip}"],
            row[f"x_{knee}"], row[f"y_{knee}"],
            row[f"x_{ankle}"], row[f"y_{ankle}"]
        )
        if i == 0:
            front_knee_extension = 0
        else:
            front_knee_extension = knee_angle - rows[-1]["knee_angle"]
        ankle_angle = calc_angle(
            row[f"x_{knee}"], row[f"y_{knee}"],
            row[f"x_{ankle}"], row[f"y_{ankle}"],
            row[f"x_{hip}"], row[f"y_{hip}"]
        )
        if i == 0:
            elbow_ang_speed = 0
            shoulder_ang_speed = 0
            hip_ang_speed = 0
            knee_ang_speed = 0
        else:
            prev = rows[-1]

            elbow_ang_speed = elbow_angle - prev["elbow_angle"]
            shoulder_ang_speed = shoulder_angle - prev["shoulder_angle"]
            hip_ang_speed = hip_angle - prev["hip_angle"]
            knee_ang_speed = knee_angle - prev["knee_angle"]

        wrist_speed = calc_speed(df, i, wrist)
        elbow_speed = calc_speed(df, i, elbow)
        shoulder_speed = calc_speed(df, i, shoulder)
        hip_speed = calc_speed(df, i, hip)
        if shoulder_speed < 1e-3:
            hip_to_shoulder_speed_ratio = 0
        else:
            hip_to_shoulder_speed_ratio = hip_speed / shoulder_speed

        if elbow_speed < 1e-3:
            shoulder_to_elbow_speed_ratio = 0
        else:
            shoulder_to_elbow_speed_ratio = shoulder_speed / elbow_speed

        if wrist_speed < 1e-3:
            elbow_to_wrist_speed_ratio = 0
        else:
            elbow_to_wrist_speed_ratio = elbow_speed / wrist_speed
        wrist_over_elbow_speed = wrist_speed / (elbow_speed + 1e-6)

        wrist_over_shoulder_speed = wrist_speed / (shoulder_speed + 1e-6)

        elbow_over_shoulder_speed = elbow_speed / (shoulder_speed + 1e-6)

        hip_over_shoulder_speed = hip_speed / (shoulder_speed + 1e-6)

        hip_over_wrist_speed = hip_speed / (wrist_speed + 1e-6)
        knee_speed = calc_speed(df, i, knee)
        ankle_speed = calc_speed(df, i, ankle)
        head_speed = calc_speed(df, i, 0)
        center_speed = calc_center_speed(df, i)
        center_x = (row["x_11"] + row["x_12"]) / 2
        center_y = (row["y_11"] + row["y_12"]) / 2
        wrist_acc = calc_acceleration(df, i, wrist)
        elbow_acc = calc_acceleration(df, i, elbow)
        shoulder_acc = calc_acceleration(df, i, shoulder)
        hip_acc = calc_acceleration(df, i, hip)
        if abs(shoulder_acc) < 1e-3:
            hip_to_shoulder_acc_ratio = 0
        else:
            hip_to_shoulder_acc_ratio = hip_acc / shoulder_acc

        if abs(elbow_acc) < 1e-3:
            shoulder_to_elbow_acc_ratio = 0
        else:
            shoulder_to_elbow_acc_ratio = shoulder_acc / elbow_acc

        if abs(wrist_acc) < 1e-3:
            elbow_to_wrist_acc_ratio = 0
        else:
            elbow_to_wrist_acc_ratio = elbow_acc / wrist_acc
        wrist_over_elbow_acc = wrist_acc / (abs(elbow_acc) + 1e-6)

        wrist_over_shoulder_acc = wrist_acc / (abs(shoulder_acc) + 1e-6)

        hip_over_shoulder_acc = hip_acc / (abs(shoulder_acc) + 1e-6)
        knee_acc = calc_acceleration(df, i, knee)
        ankle_acc = calc_acceleration(df, i, ankle)
        head_acc = calc_acceleration(df, i, 0)
        stride_length = calc_distance(
            row["x_15"], row["y_15"],
            row["x_16"], row["y_16"]
        )
        center_x = (row["x_11"] + row["x_12"]) / 2
        center_y = (row["y_11"] + row["y_12"]) / 2
        if i == 0:
            center_x_speed = 0
            center_y_speed = 0
        else:
            prev_center_x = (df.loc[i-1, "x_11"] + df.loc[i-1, "x_12"]) / 2
            prev_center_y = (df.loc[i-1, "y_11"] + df.loc[i-1, "y_12"]) / 2

            center_x_speed = center_x - prev_center_x
            center_y_speed = center_y - prev_center_y
        if i == 0:
            center_x_acc = 0
            center_y_acc = 0
        else:
            prev_center_x_speed = rows[-1]["center_x_speed"]
            prev_center_y_speed = rows[-1]["center_y_speed"]

            center_x_acc = center_x_speed - prev_center_x_speed
            center_y_acc = center_y_speed - prev_center_y_speed

        stride_length = calc_distance(
            row["x_15"], row["y_15"], row["x_16"], row["y_16"]
        )
        if i == 0:
            center_move = 0
        else:
            center_move = calc_distance(
        center_x,
        center_y,
        rows[-1]["center_x"],
        rows[-1]["center_y"]
        )
        arm_length = calc_distance(
            row[f"x_{shoulder}"],
            row[f"y_{shoulder}"],
            row[f"x_{wrist}"],
            row[f"y_{wrist}"],
        )
        hip_to_wrist = calc_distance(
            row[f"x_{hip}"], row[f"y_{hip}"], row[f"x_{wrist}"], row[f"y_{wrist}"]
        )
        head_to_center = calc_distance(row["x_0"], row["y_0"], center_x, center_y)
        shoulder_width = calc_distance(row["x_5"], row["y_5"], row["x_6"], row["y_6"])
        shoulder_width = calc_distance(row["x_5"], row["y_5"], row["x_6"], row["y_6"])
        left_right_shoulder_height = row["y_5"] - row["y_6"]

        left_right_hip_height = row["y_11"] - row["y_12"]

        left_right_knee_height = row["y_13"] - row["y_14"]

        left_right_ankle_height = row["y_15"] - row["y_16"]

        left_right_shoulder_distance = calc_distance(
            row["x_5"], row["y_5"],
            row["x_6"], row["y_6"]
        )

        left_right_hip_distance = calc_distance(
            row["x_11"], row["y_11"],
            row["x_12"], row["y_12"]
        )

        shoulder_height_diff = row["y_5"] - row["y_6"]

        hip_height_diff = row["y_11"] - row["y_12"]

        shoulder_slope = (
            (row["y_6"] - row["y_5"])
            / (row["x_6"] - row["x_5"] + 1e-6)
        )

        hip_slope = (
            (row["y_12"] - row["y_11"])
             / (row["x_12"] - row["x_11"] + 1e-6)
        )

        wrist_center_dx = row[f"x_{wrist}"] - center_x
        wrist_center_dy = row[f"y_{wrist}"] - center_y

        head_center_dx = row["x_0"] - center_x
        head_center_dy = row["y_0"] - center_y

        arm_length_ratio = arm_length / (hip_to_wrist + 1e-6)

        shoulder_width_ratio = shoulder_width / (stride_length + 1e-6)
        shoulder_tilt = np.degrees(np.arctan2(
            row["y_6"] - row["y_5"],
            row["x_6"] - row["x_5"]
        ))

        hip_tilt = np.degrees(np.arctan2(
            row["y_12"] - row["y_11"],
            row["x_12"] - row["x_11"]
        ))
        if i == 0:
            shoulder_rotation_speed = 0
        else:
            shoulder_rotation_speed = shoulder_tilt - rows[-1]["shoulder_tilt"]
        if i == 0:
            hip_rotation_speed = 0
        else:
            hip_rotation_speed = hip_tilt - rows[-1]["hip_tilt"]
        body_tilt = np.degrees(np.arctan2(
            center_y - row["y_0"],
            center_x - row["x_0"]
        ))

        wrist_radius = calc_distance(
            center_x,
            center_y,
            row[f"x_{wrist}"],
            row[f"y_{wrist}"]
        )
        if i == 0:
            wrist_path_length = 0
        else:
            wrist_path_length = rows[-1]["wrist_path_length"] + calc_distance(
                row[f"x_{wrist}"],
                row[f"y_{wrist}"],
                df.loc[i-1, f"x_{wrist}"],
                df.loc[i-1, f"y_{wrist}"]
            )
        if i == 0:
            wrist_path_speed = 0
        else:
            wrist_path_speed = calc_distance(
                row[f"x_{wrist}"],
                row[f"y_{wrist}"],
                df.loc[i-1, f"x_{wrist}"],
                df.loc[i-1, f"y_{wrist}"]
            )
        if i <= 1:
            wrist_path_curvature = 0
        else:
            dx_now = row[f"x_{wrist}"] - df.loc[i-1, f"x_{wrist}"]
            dy_now = row[f"y_{wrist}"] - df.loc[i-1, f"y_{wrist}"]

            dx_prev = df.loc[i-1, f"x_{wrist}"] - df.loc[i-2, f"x_{wrist}"]
            dy_prev = df.loc[i-1, f"y_{wrist}"] - df.loc[i-2, f"y_{wrist}"]

            angle_now = np.arctan2(dy_now, dx_now)
            angle_prev = np.arctan2(dy_prev, dx_prev)

            wrist_path_curvature = abs(angle_now - angle_prev)

        if i == 0:
            wrist_radius_speed = 0
        else:
            wrist_radius_speed = wrist_radius - rows[-1]["wrist_radius"]

        x_factor = shoulder_tilt - hip_tilt
        max_wrist_speed = max(max_wrist_speed, wrist_speed)
        max_elbow_speed = max(max_elbow_speed, elbow_speed)
        max_shoulder_speed = max(max_shoulder_speed, shoulder_speed)

        max_hip_speed = max(max_hip_speed, hip_speed)
        max_knee_speed = max(max_knee_speed, knee_speed)
        max_ankle_speed = max(max_ankle_speed, ankle_speed)
        max_wrist_acc = max(max_wrist_acc, abs(wrist_acc))
        max_elbow_acc = max(max_elbow_acc, abs(elbow_acc))
        max_shoulder_acc = max(max_shoulder_acc, abs(shoulder_acc))
        max_hip_acc = max(max_hip_acc, abs(hip_acc))
        max_knee_acc = max(max_knee_acc, abs(knee_acc))
        max_ankle_acc = max(max_ankle_acc, abs(ankle_acc))
        if i == 0:
            separation_speed = 0
        else:
            separation_speed = x_factor - rows[-1]["x_factor"]

        elbow_to_shoulder = calc_distance(
            row[f"x_{elbow}"],
            row[f"y_{elbow}"],
            row[f"x_{shoulder}"],
            row[f"y_{shoulder}"],
        )
        elbow_to_hip = calc_distance(
            row[f"x_{elbow}"], row[f"y_{elbow}"], row[f"x_{hip}"], row[f"y_{hip}"]
        )
        wrist_to_head = calc_distance(
            row[f"x_{wrist}"], row[f"y_{wrist}"], row["x_0"], row["y_0"]
        )
        release_arm_angle = calc_angle(
            row[f"x_{hip}"],
            row[f"y_{hip}"],
            row[f"x_{shoulder}"],
            row[f"y_{shoulder}"],
            row[f"x_{wrist}"],
            row[f"y_{wrist}"],
        )
        wrist_height = row[f"y_{wrist}"]
        elbow_height = row[f"y_{elbow}"]
        shoulder_height = row[f"y_{shoulder}"]
        hip_height = row[f"y_{hip}"]
        ankle_height = row[f"y_{ankle}"]
        rotation_gap = shoulder_rotation_speed - hip_rotation_speed

        arm_extension_ratio = arm_length / (shoulder_width + 1e-6)

        wrist_height_ratio = wrist_height / (shoulder_width + 1e-6)

        center_to_shoulder = calc_distance(
            center_x,
            center_y,
            row[f"x_{shoulder}"],
            row[f"y_{shoulder}"]
        )

        center_to_hip = calc_distance(
            center_x,
            center_y,
            row[f"x_{hip}"],
            row[f"y_{hip}"]
        )
        stride_ratio = stride_length / (shoulder_width + 1e-6)
        release_height_ratio = wrist_height / (shoulder_width + 1e-6)
        elbow_extension_ratio = elbow_to_shoulder / (arm_length + 1e-6)
        hip_to_head = calc_distance(
            row[f"x_{hip}"], row[f"y_{hip}"], row["x_0"], row["y_0"]
        )
        head_height = row["y_0"]
        center_height = center_y

        rows.append(
            {
                "frame": row["frame"],
                "elbow_angle": elbow_angle,
                "shoulder_angle": shoulder_angle,
                "hip_angle": hip_angle,
                "knee_angle": knee_angle,
                "front_knee_extension": front_knee_extension,
                "ankle_angle": ankle_angle,
                "wrist_speed": wrist_speed,
                "elbow_speed": elbow_speed,
                "shoulder_speed": shoulder_speed,
                "hip_speed": hip_speed,
                "hip_to_shoulder_speed_ratio": hip_to_shoulder_speed_ratio,
                "shoulder_to_elbow_speed_ratio": shoulder_to_elbow_speed_ratio,
                "elbow_to_wrist_speed_ratio": elbow_to_wrist_speed_ratio,
                "wrist_over_elbow_speed": wrist_over_elbow_speed,
                "wrist_over_shoulder_speed": wrist_over_shoulder_speed,
                "elbow_over_shoulder_speed": elbow_over_shoulder_speed,
                "hip_over_shoulder_speed": hip_over_shoulder_speed,
                "hip_over_wrist_speed": hip_over_wrist_speed,
                "knee_speed": knee_speed,
                "ankle_speed": ankle_speed,
                "head_speed": head_speed,
                "center_speed": center_speed,
                "center_x": center_x,
                "center_y": center_y,
                "wrist_path_length": wrist_path_length,
                "wrist_path_speed": wrist_path_speed,
                "wrist_path_curvature": wrist_path_curvature,
                "center_x_speed": center_x_speed,
                "center_y_speed": center_y_speed,
                "center_x_acc": center_x_acc,
                "center_y_acc": center_y_acc,
                "center_move": center_move,
                "elbow_ang_speed": elbow_ang_speed,
                "shoulder_ang_speed": shoulder_ang_speed,
                "hip_ang_speed": hip_ang_speed,
                "knee_ang_speed": knee_ang_speed,
                "wrist_acc": wrist_acc,
                "elbow_acc": elbow_acc,
                "shoulder_acc": shoulder_acc,
                "hip_acc": hip_acc,
                

                "wrist_over_elbow_acc": wrist_over_elbow_acc,
                "wrist_over_shoulder_acc": wrist_over_shoulder_acc,
                "hip_over_shoulder_acc": hip_over_shoulder_acc,
                "knee_acc": knee_acc,
                "ankle_acc": ankle_acc,
                "head_acc": head_acc,
                "stride_length": stride_length,
                "arm_length": arm_length,
                "hip_to_wrist": hip_to_wrist,
                "head_to_center": head_to_center,
                "shoulder_width": shoulder_width,
                "left_right_shoulder_height": left_right_shoulder_height,
                "left_right_hip_height": left_right_hip_height,
                "left_right_knee_height": left_right_knee_height,
                "left_right_ankle_height": left_right_ankle_height,
                "left_right_shoulder_distance": left_right_shoulder_distance,
                "left_right_hip_distance": left_right_hip_distance,
                "shoulder_height_diff": shoulder_height_diff,
                "hip_height_diff": hip_height_diff,
                "shoulder_slope": shoulder_slope,
                "hip_slope": hip_slope,
                "wrist_center_dx": wrist_center_dx,
                "wrist_center_dy": wrist_center_dy,
                "head_center_dx": head_center_dx,
                "head_center_dy": head_center_dy,
                "arm_length_ratio": arm_length_ratio,
                "shoulder_width_ratio": shoulder_width_ratio,
                "shoulder_height_diff": shoulder_height_diff,
                "hip_height_diff": hip_height_diff,
                "shoulder_slope": shoulder_slope,
                "hip_slope": hip_slope,
                "wrist_center_dx": wrist_center_dx,
                "wrist_center_dy": wrist_center_dy,
                "head_center_dx": head_center_dx,
                "head_center_dy": head_center_dy,
                "arm_length_ratio": arm_length_ratio,
                "shoulder_width_ratio": shoulder_width_ratio,
                "shoulder_tilt": shoulder_tilt,
                "hip_tilt": hip_tilt,
                "body_tilt": body_tilt,
                "rotation_gap": rotation_gap,
                "arm_extension_ratio": arm_extension_ratio,
                "wrist_height_ratio": wrist_height_ratio,
                "center_to_shoulder": center_to_shoulder,
                "center_to_hip": center_to_hip,
                "wrist_radius": wrist_radius,
                "wrist_radius_speed": wrist_radius_speed,
                "shoulder_rotation_speed": shoulder_rotation_speed,
                "hip_rotation_speed": hip_rotation_speed,
                "x_factor": x_factor,
                "separation_speed": separation_speed,
                "elbow_to_shoulder": elbow_to_shoulder,
                "elbow_to_hip": elbow_to_hip,
                "wrist_to_head": wrist_to_head,
                "release_arm_angle": release_arm_angle,
                "wrist_height": wrist_height,
                "elbow_height": elbow_height,
                "shoulder_height": shoulder_height,
                "hip_height": hip_height,
                "ankle_height": ankle_height,
                "stride_ratio": stride_ratio,
                "release_height_ratio": release_height_ratio,
                "elbow_extension_ratio": elbow_extension_ratio,
                "hip_to_head": hip_to_head,
                "head_height": head_height,
                "center_height": center_height,
            }
        )
    features = pd.DataFrame(rows)
    std_elbow_angle = features["elbow_angle"].std()
    std_shoulder_angle = features["shoulder_angle"].std()
    std_hip_angle = features["hip_angle"].std()
    std_x_factor = features["x_factor"].std()
    std_body_tilt = features["body_tilt"].std()
    std_rotation_gap = features["rotation_gap"].std()
    features["std_elbow_angle"] = std_elbow_angle
    features["std_shoulder_angle"] = std_shoulder_angle
    features["std_hip_angle"] = std_hip_angle
    features["std_x_factor"] = std_x_factor
    features["std_body_tilt"] = std_body_tilt
    features["std_rotation_gap"] = std_rotation_gap
    release = detect_release(hand)

    # detect_releaseで得たフレームが
    # features.csvに存在しない場合、一番近いフレームを使う
    available_frames = features["frame"].dropna()

    if len(available_frames) == 0:
        print("特徴量データにフレームがありません")
        return

    if release not in available_frames.values:
        nearest_frame = available_frames.iloc[
            (available_frames - release).abs().argmin()
        ]

        print(
            f"リリースフレーム {release} は特徴量に存在しないため、"
            f"最も近いフレーム {nearest_frame} を使用します"
        )

        release = int(nearest_frame)

    print(f"特徴量で使用するリリースフレーム: {release}")

    # releaseフレームの5フレーム前～releaseフレームを取得
    pre5 = features[
        (features["frame"] >= release - 5) &
        (features["frame"] <= release)
    ]
    release_time = release / 30
    release_ratio = release / len(df)

    idx = features[features["frame"] == release].index
    print("DEBUG release =", release)
    print("DEBUG frame min =", features["frame"].min())
    print("DEBUG frame max =", features["frame"].max())
    print("DEBUG release exists =", (features["frame"] == release).any())
    if len(idx) == 0:
        print(f"警告: リリースフレーム {release} が特徴量データに存在しません")
        print(f"特徴量のフレーム範囲: {features['frame'].min()} ～ {features['frame'].max()}")
        return

    idx = idx[0]

    pre10_start = max(0, idx - 10)
    release_wrist_speed_change = (
        features.loc[idx, "wrist_speed"]
        - features.loc[pre10_start, "wrist_speed"]
    )

    release_elbow_speed_change = (
        features.loc[idx, "elbow_speed"]
        - features.loc[pre10_start, "elbow_speed"]
    )

    release_shoulder_speed_change = (
        features.loc[idx, "shoulder_speed"]
        - features.loc[pre10_start, "shoulder_speed"]
    )

    release_hip_speed_change = (
        features.loc[idx, "hip_speed"]
        - features.loc[pre10_start, "hip_speed"]
    )

    release_x_factor_change = (
        features.loc[idx, "x_factor"]
        - features.loc[pre10_start, "x_factor"]
    )

    release_body_tilt_change = (
        features.loc[idx, "body_tilt"]
        - features.loc[pre10_start, "body_tilt"]
    )
    pre10_start = max(0, idx - 10)
    pre10 = features.iloc[pre10_start:idx]

    pre10_wrist_speed = pre10["wrist_speed"].mean()
    pre10_elbow_speed = pre10["elbow_speed"].mean()
    pre10_shoulder_speed = pre10["shoulder_speed"].mean()
    pre10_x_factor = pre10["x_factor"].mean()
    pre10_body_tilt = pre10["body_tilt"].mean()
    pre10_rotation_gap = pre10["rotation_gap"].mean()
    hip_peak_idx = features["hip_speed"].idxmax()
    shoulder_peak_idx = features["shoulder_speed"].idxmax()
    elbow_peak_idx = features["elbow_speed"].idxmax()
    wrist_peak_idx = features["wrist_speed"].idxmax()

    hip_peak_frame = features.loc[hip_peak_idx, "frame"]
    shoulder_peak_frame = features.loc[shoulder_peak_idx, "frame"]
    elbow_peak_frame = features.loc[elbow_peak_idx, "frame"]
    wrist_peak_frame = features.loc[wrist_peak_idx, "frame"]
    print("ピークフレーム")
    print("hip:", hip_peak_frame)
    print("shoulder:", shoulder_peak_frame)
    print("elbow:", elbow_peak_frame)
    print("wrist:", wrist_peak_frame)
    hip_to_shoulder_peak_delay = shoulder_peak_frame - hip_peak_frame
    shoulder_to_elbow_peak_delay = elbow_peak_frame - shoulder_peak_frame
    elbow_to_wrist_peak_delay = wrist_peak_frame - elbow_peak_frame
    hip_to_wrist_peak_delay = wrist_peak_frame - hip_peak_frame
    hip_peak_speed = features.loc[hip_peak_idx, "hip_speed"]
    shoulder_peak_speed = features.loc[shoulder_peak_idx, "shoulder_speed"]
    elbow_peak_speed = features.loc[elbow_peak_idx, "elbow_speed"]
    wrist_peak_speed = features.loc[wrist_peak_idx, "wrist_speed"]
    max_wrist_speed_idx = features["wrist_speed"].idxmax()
    max_elbow_speed_idx = features["elbow_speed"].idxmax()
    max_shoulder_speed_idx = features["shoulder_speed"].idxmax()
    max_hip_speed_idx = features["hip_speed"].idxmax()
    max_knee_speed_idx = features["knee_speed"].idxmax()
    max_ankle_speed_idx = features["ankle_speed"].idxmax()
    max_wrist_acc_idx = features["wrist_acc"].abs().idxmax()
    max_elbow_acc_idx = features["elbow_acc"].abs().idxmax()
    max_shoulder_acc_idx = features["shoulder_acc"].abs().idxmax()

    max_wrist_speed_frame = features.loc[max_wrist_speed_idx, "frame"]
    max_elbow_speed_frame = features.loc[max_elbow_speed_idx, "frame"]
    max_shoulder_speed_frame = features.loc[max_shoulder_speed_idx, "frame"]
    max_hip_speed_frame = features.loc[max_hip_speed_idx, "frame"]
    max_knee_speed_frame = features.loc[max_knee_speed_idx, "frame"]
    max_ankle_speed_frame = features.loc[max_ankle_speed_idx, "frame"]

    max_wrist_acc_frame = features.loc[max_wrist_acc_idx, "frame"]
    max_elbow_acc_frame = features.loc[max_elbow_acc_idx, "frame"]
    max_shoulder_acc_frame = features.loc[max_shoulder_acc_idx, "frame"]

    hip_to_shoulder_delay = shoulder_peak_frame - hip_peak_frame
    shoulder_to_elbow_delay = elbow_peak_frame - shoulder_peak_frame

    # releaseフレームの直前5行～release行を取得
    release_idx = features.index[
        (features["frame"] - release).abs().argmin()
    ]

    pre5_start = max(0, release_idx - 5)

    pre5 = features.iloc[pre5_start:release_idx + 1]

    pre5_wrist_speed = pre5["wrist_speed"].mean()
    pre5_elbow_speed = pre5["elbow_speed"].mean()
    pre5_shoulder_speed = pre5["shoulder_speed"].mean()
    pre5_hip_speed = pre5["hip_speed"].mean()

    pre5_wrist_acc = pre5["wrist_acc"].mean()

    pre5_x_factor = pre5["x_factor"].mean()

    pre5_rotation_gap = pre5["rotation_gap"].mean()
    pre5_wrist_speed_max = pre5["wrist_speed"].max()
    pre5_wrist_speed_min = pre5["wrist_speed"].min()
    pre5_wrist_speed_std = pre5["wrist_speed"].std()
    pre5_shoulder_speed_max = pre5["shoulder_speed"].max()
    pre5_shoulder_speed_min = pre5["shoulder_speed"].min()
    pre5_shoulder_speed_std = pre5["shoulder_speed"].std()
    pre5_hip_speed_max = pre5["hip_speed"].max()
    pre5_hip_speed_min = pre5["hip_speed"].min()
    pre5_hip_speed_std = pre5["hip_speed"].std()
    pre5_x_factor_max = pre5["x_factor"].max()
    pre5_x_factor_min = pre5["x_factor"].min()
    pre5_x_factor_std = pre5["x_factor"].std()
    pre5_wrist_acc_max = pre5["wrist_acc"].max()
    pre5_wrist_acc_min = pre5["wrist_acc"].min()
    pre5_wrist_acc_std = pre5["wrist_acc"].std()

    release_time = release / 30 
    release_ratio = release / len(df)
    idx = features[features["frame"] == release].index

    if len(idx) > 0:
        idx = idx[0]
        features.loc[idx, "release_minus_pre10_wrist_speed"] = (
            features.loc[idx, "wrist_speed"] - pre10_wrist_speed
        )

        features.loc[idx, "release_minus_pre10_elbow_speed"] = (
            features.loc[idx, "elbow_speed"] - pre10_elbow_speed
        )

        features.loc[idx, "release_minus_pre10_shoulder_speed"] = (
            features.loc[idx, "shoulder_speed"] - pre10_shoulder_speed
        )

        features.loc[idx, "release_minus_pre10_x_factor"] = (
            features.loc[idx, "x_factor"] - pre10_x_factor
        )

        features.loc[idx, "release_minus_pre10_body_tilt"] = (
            features.loc[idx, "body_tilt"] - pre10_body_tilt
        )

        features.loc[idx, "release_minus_pre10_rotation_gap"] = (
            features.loc[idx, "rotation_gap"] - pre10_rotation_gap
        )

        features.loc[idx, "max_wrist_speed_frame"] = max_wrist_speed_frame
        features.loc[idx, "max_elbow_speed_frame"] = max_elbow_speed_frame
        features.loc[idx, "max_shoulder_speed_frame"] = max_shoulder_speed_frame
        features.loc[idx, "max_hip_speed_frame"] = max_hip_speed_frame
        features.loc[idx, "max_knee_speed_frame"] = max_knee_speed_frame
        features.loc[idx, "max_ankle_speed_frame"] = max_ankle_speed_frame

        features.loc[idx, "max_wrist_acc_frame"] = max_wrist_acc_frame
        features.loc[idx, "max_elbow_acc_frame"] = max_elbow_acc_frame
        features.loc[idx, "max_shoulder_acc_frame"] = max_shoulder_acc_frame

        features.loc[idx, "release_elbow_angle"] = features.loc[idx, "elbow_angle"]
        features.loc[idx, "release_shoulder_angle"] = features.loc[idx, "shoulder_angle"]
        features.loc[idx, "release_hip_angle"] = features.loc[idx, "hip_angle"]
        features.loc[idx, "release_knee_angle"] = features.loc[idx, "knee_angle"]
        features.loc[idx, "release_ankle_angle"] = features.loc[idx, "ankle_angle"]

        features.loc[idx, "release_wrist_speed"] = features.loc[idx, "wrist_speed"]
        features.loc[idx, "release_elbow_speed"] = features.loc[idx, "elbow_speed"]
        features.loc[idx, "release_shoulder_speed"] = features.loc[idx, "shoulder_speed"]
        features.loc[idx, "release_hip_speed"] = features.loc[idx, "hip_speed"]
        features.loc[idx, "release_x_factor"] = features.loc[idx, "x_factor"]

        features.loc[idx, "release_time"] = release_time
        features.loc[idx, "release_center_x"] = features.loc[idx, "center_x"]
        features.loc[idx, "release_center_y"] = features.loc[idx, "center_y"]

        features.loc[idx, "release_wrist_radius"] = features.loc[idx, "wrist_radius"]
        features.loc[idx, "release_wrist_radius_speed"] = features.loc[
            idx, "wrist_radius_speed"
        ]

        features.loc[idx, "pre5_wrist_speed"] = pre5_wrist_speed
        features.loc[idx, "pre5_elbow_speed"] = pre5_elbow_speed
        features.loc[idx, "pre5_shoulder_speed"] = pre5_shoulder_speed
        features.loc[idx, "pre5_hip_speed"] = pre5_hip_speed

        features.loc[idx, "pre5_wrist_acc"] = pre5_wrist_acc

        features.loc[idx, "pre5_x_factor"] = pre5_x_factor

        features.loc[idx, "pre5_rotation_gap"] = pre5_rotation_gap
        features.loc[idx, "pre5_wrist_speed_max"] = pre5_wrist_speed_max
        features.loc[idx, "pre5_wrist_speed_min"] = pre5_wrist_speed_min
        features.loc[idx, "pre5_wrist_speed_std"] = pre5_wrist_speed_std

        features.loc[idx, "pre5_shoulder_speed_max"] = pre5_shoulder_speed_max
        features.loc[idx, "pre5_shoulder_speed_min"] = pre5_shoulder_speed_min
        features.loc[idx, "pre5_shoulder_speed_std"] = pre5_shoulder_speed_std

        features.loc[idx, "pre5_hip_speed_max"] = pre5_hip_speed_max
        features.loc[idx, "pre5_hip_speed_min"] = pre5_hip_speed_min
        features.loc[idx, "pre5_hip_speed_std"] = pre5_hip_speed_std

        features.loc[idx, "pre5_x_factor_max"] = pre5_x_factor_max
        features.loc[idx, "pre5_x_factor_min"] = pre5_x_factor_min
        features.loc[idx, "pre5_x_factor_std"] = pre5_x_factor_std

        features.loc[idx, "pre5_wrist_acc_max"] = pre5_wrist_acc_max
        features.loc[idx, "pre5_wrist_acc_min"] = pre5_wrist_acc_min
        features.loc[idx, "pre5_wrist_acc_std"] = pre5_wrist_acc_std

        features.loc[idx, "max_wrist_speed"] = max_wrist_speed
        features.loc[idx, "max_elbow_speed"] = max_elbow_speed
        features.loc[idx, "max_shoulder_speed"] = max_shoulder_speed

        features.loc[idx, "max_hip_speed"] = max_hip_speed
        features.loc[idx, "max_knee_speed"] = max_knee_speed
        features.loc[idx, "max_ankle_speed"] = max_ankle_speed
        features.loc[idx, "release_ratio"] = release_ratio

        features.loc[idx, "hip_peak_frame"] = hip_peak_frame
        features.loc[idx, "shoulder_peak_frame"] = shoulder_peak_frame
        features.loc[idx, "elbow_peak_frame"] = elbow_peak_frame
        features.loc[idx, "wrist_peak_frame"] = wrist_peak_frame

        features.loc[idx, "hip_to_shoulder_peak_delay"] = hip_to_shoulder_peak_delay
        features.loc[idx, "shoulder_to_elbow_peak_delay"] = shoulder_to_elbow_peak_delay
        features.loc[idx, "elbow_to_wrist_peak_delay"] = elbow_to_wrist_peak_delay
        features.loc[idx, "hip_to_wrist_peak_delay"] = hip_to_wrist_peak_delay

        features.loc[idx, "hip_peak_speed"] = hip_peak_speed
        features.loc[idx, "shoulder_peak_speed"] = shoulder_peak_speed
        features.loc[idx, "elbow_peak_speed"] = elbow_peak_speed
        features.loc[idx, "wrist_peak_speed"] = wrist_peak_speed

        features.loc[idx, "hip_to_shoulder_delay"] = hip_to_shoulder_delay
        features.loc[idx, "shoulder_to_elbow_delay"] = shoulder_to_elbow_delay
        features.loc[idx, "max_wrist_acc"] = max_wrist_acc
        features.loc[idx, "max_elbow_acc"] = max_elbow_acc
        features.loc[idx, "max_shoulder_acc"] = max_shoulder_acc
        features.loc[idx, "max_hip_acc"] = max_hip_acc
        features.loc[idx, "max_knee_acc"] = max_knee_acc
        features.loc[idx, "max_ankle_acc"] = max_ankle_acc
        features.loc[idx, "pre5_wrist_speed"] = pre5_wrist_speed
        features.loc[idx, "pre5_shoulder_speed"] = pre5_shoulder_speed
        features.loc[idx, "pre5_hip_speed"] = pre5_hip_speed
        features.loc[idx, "pre5_x_factor"] = pre5_x_factor
        features.loc[idx, "pre5_wrist_acc"] = pre5_wrist_acc
        features.loc[idx, "release_wrist_speed_change"] = release_wrist_speed_change
        features.loc[idx, "release_elbow_speed_change"] = release_elbow_speed_change
        features.loc[idx, "release_shoulder_speed_change"] = release_shoulder_speed_change
        features.loc[idx, "release_hip_speed_change"] = release_hip_speed_change
        features.loc[idx, "release_x_factor_change"] = release_x_factor_change
        features.loc[idx, "release_body_tilt_change"] = release_body_tilt_change
    Path("output").mkdir(exist_ok=True)

    features.to_csv("output/features.csv", index=False)

    print("特徴量CSVを保存しました！")

    return release
