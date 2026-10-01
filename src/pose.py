from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n-pose.pt")

def analyze_video(video_path):

    print("現在の作業フォルダ:", Path.cwd())
    print("動画の絶対パス:", Path(video_path).resolve())
    print("動画は存在する？", Path(video_path).exists())

    results = list(model.predict(
    source=video_path,
    save=False,
    conf=0.3,
    stream=True,
    verbose=False
))
    return results