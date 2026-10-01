import pandas as pd
from pathlib import Path


def save_dataset(ball_speed, release_frame, video_name):

    features = pd.read_csv("output/features.csv")

    # frame番号でリリースフレームを検索
    row = features[features["frame"] == release_frame]

    if len(row) == 0:
        print("リリースフレームが見つかりませんでした")
        return

    row = row.iloc[0].copy()
    print("===== dataset保存直前 =====")
    print("release_frame:", release_frame)
    print("frame:", row["frame"])
    print("pre5_wrist_speed:", row["pre5_wrist_speed"])
    print("pre5_elbow_speed:", row["pre5_elbow_speed"])
    print("pre5_shoulder_speed:", row["pre5_shoulder_speed"])
    print("pre5_hip_speed:", row["pre5_hip_speed"])
    print("pre5_wrist_acc:", row["pre5_wrist_acc"])
    print("pre5_x_factor:", row["pre5_x_factor"])
    print("pre5_rotation_gap:", row["pre5_rotation_gap"])
    print("==========================")
    keypoints = pd.read_csv("output/keypoints.csv")

    release = keypoints[keypoints["frame"] == release_frame]

    if len(release) > 0:

        release = release.iloc[0]

        # ===== リリース位置 =====

        # 手首
        row["release_wrist_x"] = release["x_9"]
        row["release_wrist_y"] = release["y_9"]

        # 肘
        row["release_elbow_x"] = release["x_7"]
        row["release_elbow_y"] = release["y_7"]

        # 肩
        row["release_shoulder_x"] = release["x_5"]
        row["release_shoulder_y"] = release["y_5"]

        # 腰
        row["release_hip_x"] = release["x_11"]
        row["release_hip_y"] = release["y_11"]

        # 膝
        row["release_knee_x"] = release["x_13"]
        row["release_knee_y"] = release["y_13"]

        # 足首
        row["release_ankle_x"] = release["x_15"]
        row["release_ankle_y"] = release["y_15"]

    row["ball_speed"] = ball_speed
    row["video_name"] = video_name
    save_path = Path("output/dataset.csv")

    if save_path.exists():

        old = pd.read_csv(save_path)

        # video_name列が存在する場合
        if "video_name" in old.columns:

            # 同じ動画がすでに登録されているか確認
            if video_name in old["video_name"].values:
                print(f"すでに登録済みの動画です: {video_name}")
                return

        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)

    else:

        new = pd.DataFrame([row])

    new.to_csv(save_path, index=False)

    print("dataset.csv に追加しました！")
    print("現在のデータ数:", len(new))
