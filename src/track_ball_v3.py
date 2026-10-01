import cv2
import csv
import math
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/runs/detect/ball_finetune_mydata_v2/weights/best.pt")
VIDEO_LIST = [
    Path("videos/pitch_102_1.mp4"),
    Path("videos/pitch_98_1.mp4"),
    Path("videos/pitch_98_2.mp4"),
]
CSV_DIR = Path("output/ball_tracking_v3")
VIDEO_DIR = Path("runs/detect/ball_tracking_v3")
CSV_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
CONF_THRESHOLD = 0.20
MAX_MISSING = 3
MAX_MOVE = 80.0
MIN_TRACK_LENGTH = 5


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)


def predict(points):
    if len(points) < 2:
        return points[-1]["x"], points[-1]["y"]
    return (
        points[-1]["x"] + (points[-1]["x"] - points[-2]["x"]),
        points[-1]["y"] + (points[-1]["y"] - points[-2]["y"]),
    )


def choose(dets, pred):
    if not dets:
        return None
    best = min(dets, key=lambda d: dist(d["center"], pred))
    return best if dist(best["center"], pred) <= MAX_MOVE else None


def track_video(model, video_path):
    print("\n" + "=" * 60)
    print("動画:", video_path)
    print("=" * 60)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("動画を開けません")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS:", fps)
    print("総フレーム数:", total)
    raw = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                dets.append({"center": center(xyxy), "conf": conf, "xyxy": xyxy})
        raw.append(dets)
        fi += 1
    cap.release()

    tracks = []
    for start, dets in enumerate(raw):
        if not dets:
            continue
        first = max(dets, key=lambda d: d["conf"])
        pts = [
            {
                "frame": start,
                "x": first["center"][0],
                "y": first["center"][1],
                "conf": first["conf"],
                "source": "detected",
            }
        ]
        missing = 0
        for j in range(start + 1, len(raw)):
            pred = predict(pts)
            cand = choose(raw[j], pred)
            if cand:
                pts.append(
                    {
                        "frame": j,
                        "x": cand["center"][0],
                        "y": cand["center"][1],
                        "conf": cand["conf"],
                        "source": "detected",
                    }
                )
                missing = 0
            elif missing < MAX_MISSING:
                pts.append(
                    {
                        "frame": j,
                        "x": pred[0],
                        "y": pred[1],
                        "conf": 0.0,
                        "source": "predicted",
                    }
                )
                missing += 1
            else:
                break
        if len(pts) >= MIN_TRACK_LENGTH:
            tracks.append(pts)

    if not tracks:
        print("有効な軌跡なし")
        return

    def score(p):
        mc = sum(x["conf"] for x in p) / len(p)
        dd = dist((p[0]["x"], p[0]["y"]), (p[-1]["x"], p[-1]["y"]))
        return (len(p), mc, dd)

    tracks.sort(key=score, reverse=True)
    pts = tracks[0]
    mc = sum(p["conf"] for p in pts) / len(pts)
    dd = dist((pts[0]["x"], pts[0]["y"]), (pts[-1]["x"], pts[-1]["y"]))
    pred_count = sum(p["source"] == "predicted" for p in pts)
    print("\n採用軌跡")
    print(f"Frame: {pts[0]['frame']} → {pts[-1]['frame']}")
    print(f"追跡点数: {len(pts)}")
    print(f"平均confidence: {mc:.3f}")
    print(f"移動距離: {dd:.2f}px")
    print(f"予測補完点数: {pred_count}")

    csv_path = CSV_DIR / f"{video_path.stem}_tracking.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "confidence", "source"])
        for p in pts:
            w.writerow(
                [
                    p["frame"],
                    f"{p['x']:.3f}",
                    f"{p['y']:.3f}",
                    f"{p['conf']:.6f}",
                    p["source"],
                ]
            )

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = VIDEO_DIR / f"{video_path.stem}_tracking.mp4"
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    pmap = {p["frame"]: p for p in pts}
    history = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in pmap:
            p = pmap[fi]
            xy = (int(round(p["x"])), int(round(p["y"])))
            cv2.circle(
                frame,
                xy,
                7,
                (0, 255, 255) if p["source"] == "predicted" else (0, 0, 255),
                -1,
            )
            history.append(xy)
        for a, b in zip(history[:-1], history[1:]):
            cv2.line(frame, a, b, (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Frame: {fi}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()
    print("CSV:", csv_path)
    print("動画:", out)


def main():
    print("=" * 60)
    print("ボール追跡 v3")
    print("最大3フレームの検出欠損を予測補完")
    print("=" * 60)
    print("モデル:", MODEL_PATH)
    if not MODEL_PATH.exists():
        print("モデルがありません")
        return
    model = YOLO(str(MODEL_PATH))
    print("モデル読み込み完了")
    for v in VIDEO_LIST:
        if v.exists():
            track_video(model, v)
        else:
            print("動画がありません:", v)
    print("\n" + "=" * 60)
    print("全動画の追跡完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
