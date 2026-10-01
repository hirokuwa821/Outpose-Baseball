from pathlib import Path
import csv

TRACKING_DIR = Path("output/ball_tracking_v2")

print("=" * 70)
print("追跡結果 v2")
print("=" * 70)

files = sorted(TRACKING_DIR.glob("*_tracking.csv"))

print("追跡CSV数:", len(files))
print()

for csv_file in files:

    with open(csv_file, "r", encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(csv_file.name)
        print("  → 追跡データなし")
        print()
        continue

    start_frame = int(rows[0]["frame"])
    end_frame = int(rows[-1]["frame"])
    num_frames = len(rows)

    print(csv_file.name)
    print(f"  フレーム: {start_frame} ～ {end_frame}")
    print(f"  追跡フレーム数: {num_frames}")
    print()

print("=" * 70)
print("確認完了")
print("=" * 70)

