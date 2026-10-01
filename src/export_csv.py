import pandas as pd
from pathlib import Path


def export_keypoints(results):

    rows = []

    for frame_num, result in enumerate(results):

        if result.keypoints is None:
            continue

        # 人物が1人も検出されなかった場合はスキップ
        if len(result.keypoints.xy) == 0:
            continue

        points = result.keypoints.xy[0]

        row = {"frame": frame_num}

        for i, point in enumerate(points):
            row[f"x_{i}"] = float(point[0])
            row[f"y_{i}"] = float(point[1])

        rows.append(row)

    df = pd.DataFrame(rows)

    Path("output").mkdir(exist_ok=True)

    save_path = Path("output") / "keypoints.csv"

    df.to_csv(save_path, index=False)

    print("CSV保存成功！")
    print(save_path.resolve())
