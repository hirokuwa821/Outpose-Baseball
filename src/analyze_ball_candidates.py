import csv
import math
from pathlib import Path
from collections import defaultdict

# ============================================================
# 設定
# ============================================================

INPUT_CSV = "output/ball_motion_candidates_v2.csv"

# 実際の投球付近
START_FRAME = 40
END_FRAME = 75

# フレーム間の最大移動距離
MAX_MOVE = 150

# 最低連続フレーム数
MIN_LENGTH = 5

# 上位何本表示するか
TOP_N = 20


# ============================================================
# CSV読み込み
# ============================================================

candidates_by_frame = defaultdict(list)

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        frame = int(row["frame"])

        if frame < START_FRAME or frame > END_FRAME:
            continue

        candidates_by_frame[frame].append(
            {
                "frame": frame,
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "score": float(row["score"]),
                "area": float(row["area"]),
                "circularity": float(row["circularity"]),
            }
        )


# ============================================================
# 結果
# ============================================================

print()
print("==========================================")
print("ボール候補追跡解析")
print("==========================================")

print("解析範囲:", START_FRAME, "～", END_FRAME)

print()


# ============================================================
# 候補数確認
# ============================================================

for frame in range(START_FRAME, END_FRAME + 1):

    count = len(candidates_by_frame.get(frame, []))

    print(f"Frame {frame:3d}: {count:2d} candidates")


# ============================================================
# 連続軌跡を作る
# ============================================================

tracks = []


for start_frame in range(START_FRAME, END_FRAME + 1):

    candidates = candidates_by_frame.get(start_frame, [])

    for candidate in candidates:

        track = [candidate]

        previous = candidate

        for frame in range(start_frame + 1, END_FRAME + 1):

            next_candidates = candidates_by_frame.get(frame, [])

            if not next_candidates:
                break

            # --------------------------------------------
            # 一番近い候補
            # --------------------------------------------

            best = None
            best_distance = float("inf")

            for c in next_candidates:

                d = math.sqrt(
                    (c["x"] - previous["x"]) ** 2 + (c["y"] - previous["y"]) ** 2
                )

                if d < best_distance:

                    best_distance = d
                    best = c

            if best is None:
                break

            if best_distance > MAX_MOVE:
                break

            track.append(best)

            previous = best

        if len(track) >= MIN_LENGTH:

            tracks.append(track)


# ============================================================
# 重複除去
# ============================================================

unique_tracks = []

seen = set()


for track in tracks:

    key = tuple((c["frame"], round(c["x"], 1), round(c["y"], 1)) for c in track)

    if key in seen:
        continue

    seen.add(key)

    unique_tracks.append(track)


# ============================================================
# 軌跡評価
# ============================================================

results = []


for track in unique_tracks:

    length = len(track)

    total_distance = 0.0

    dx_total = 0.0
    dy_total = 0.0

    movements = []

    for i in range(1, length):

        dx = track[i]["x"] - track[i - 1]["x"]

        dy = track[i]["y"] - track[i - 1]["y"]

        d = math.sqrt(dx * dx + dy * dy)

        movements.append(d)

        total_distance += d

        dx_total += dx
        dy_total += dy

    if length > 1:

        average_move = total_distance / (length - 1)

    else:

        average_move = 0

    # --------------------------------------------------------
    # 直線距離
    # --------------------------------------------------------

    straight_distance = math.sqrt(dx_total**2 + dy_total**2)

    if total_distance > 0:

        straightness = straight_distance / total_distance

    else:

        straightness = 0

    # --------------------------------------------------------
    # 動きのばらつき
    # --------------------------------------------------------

    if movements:

        mean_move = sum(movements) / len(movements)

        variance = sum((x - mean_move) ** 2 for x in movements) / len(movements)

        std_move = math.sqrt(variance)

    else:

        std_move = 0

    # --------------------------------------------------------
    # スコア平均
    # --------------------------------------------------------

    mean_score = sum(c["score"] for c in track) / length

    # --------------------------------------------------------
    # 評価スコア
    # --------------------------------------------------------

    evaluation = 0

    # 長い連続軌跡を評価
    evaluation += length * 10

    # 直線的な動きを評価
    evaluation += straightness * 100

    # YOLOスコア
    evaluation += mean_score * 30

    # 極端な速度変化を少し減点
    evaluation -= std_move * 0.1

    results.append(
        {
            "start": track[0]["frame"],
            "end": track[-1]["frame"],
            "length": length,
            "start_x": track[0]["x"],
            "start_y": track[0]["y"],
            "end_x": track[-1]["x"],
            "end_y": track[-1]["y"],
            "distance": total_distance,
            "straightness": straightness,
            "mean_move": mean_move,
            "std_move": std_move,
            "mean_score": mean_score,
            "evaluation": evaluation,
            "track": track,
        }
    )


# ============================================================
# 評価順
# ============================================================

results.sort(key=lambda x: x["evaluation"], reverse=True)


# ============================================================
# 表示
# ============================================================

print()
print("==========================================")
print("有力軌跡 TOP", TOP_N)
print("==========================================")


for i, result in enumerate(results[:TOP_N], 1):

    print()
    print(f"[{i}] " f"Frame {result['start']} → " f"{result['end']}")

    print(f"    length       : " f"{result['length']}")

    print(
        f"    start        : " f"({result['start_x']:.1f}, " f"{result['start_y']:.1f})"
    )

    print(f"    end          : " f"({result['end_x']:.1f}, " f"{result['end_y']:.1f})")

    print(f"    distance     : " f"{result['distance']:.2f}px")

    print(f"    straightness : " f"{result['straightness']:.3f}")

    print(f"    mean move    : " f"{result['mean_move']:.2f}px")

    print(f"    mean score   : " f"{result['mean_score']:.3f}")

    print(f"    evaluation   : " f"{result['evaluation']:.2f}")


# ============================================================
# CSV保存
# ============================================================

output_csv = "output/" "ball_candidate_tracks.csv"

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "rank",
            "start_frame",
            "end_frame",
            "length",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "distance_px",
            "straightness",
            "mean_move_px",
            "std_move_px",
            "mean_confidence",
            "evaluation",
        ]
    )

    for rank, result in enumerate(results, 1):

        writer.writerow(
            [
                rank,
                result["start"],
                result["end"],
                result["length"],
                result["start_x"],
                result["start_y"],
                result["end_x"],
                result["end_y"],
                result["distance"],
                result["straightness"],
                result["mean_move"],
                result["std_move"],
                result["mean_score"],
                result["evaluation"],
            ]
        )


print()
print("==========================================")
print("解析完了")
print("==========================================")

print("軌跡数:", len(results))

print("保存先:", output_csv)
