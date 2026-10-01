import numpy as np


def calculate_velocity_robust(data, conf_threshold=0.5, use_real_only=True):
    # 1. フィルタリング: confidence閾値 + 補間点除外
    filtered = [
        d
        for d in data
        if d["confidence"] >= conf_threshold
        and (not use_real_only or d["interpolated"] == 0)
    ]

    if len(filtered) < 3:
        return None  # データ不足

    # 2. 隣接差分で速度を計算(フィルタ後のデータで)
    velocities = []
    for i in range(1, len(filtered)):
        p1, p2 = filtered[i - 1], filtered[i]
        dx = p2["cx"] - p1["cx"]
        dy = p2["cy"] - p1["cy"]
        dist = math.sqrt(dx * dx + dy * dy)
        dt = p2["time_sec"] - p1["time_sec"]
        if dt <= 0:
            continue
        velocities.append(dist / dt)

    if not velocities:
        return None

    # 3. 外れ値除去(中央値からのズレが大きい区間を除外)
    median_v = np.median(velocities)
    mad = np.median(
        [abs(v - median_v) for v in velocities]
    )  # median absolute deviation
    threshold = median_v + 3 * mad * 1.4826  # 3σ相当

    clean_velocities = [v for v in velocities if v <= threshold]

    # 4. 代表速度: 線形回帰の傾きで求める(推奨)
    times = np.array([d["time_sec"] for d in filtered])
    positions = np.array([[d["cx"], d["cy"]] for d in filtered])
    # 位置の大きさ(原点からの距離)ではなく、進行方向の距離を積算するか、
    # 単純に累積距離を時間で線形フィットするのが安定
    cumulative_dist = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    )
    slope, intercept = np.polyfit(times, cumulative_dist, 1)  # slope = px/s

    return {
        "velocity_px_s_regression": slope,  # 線形回帰による代表速度(推奨)
        "velocity_px_s_median": median_v,  # 外れ値除去前の中央値
        "velocity_px_s_clean_max": max(clean_velocities) if clean_velocities else None,
        "n_points_used": len(filtered),
        "n_outliers_removed": len(velocities) - len(clean_velocities),
    }
