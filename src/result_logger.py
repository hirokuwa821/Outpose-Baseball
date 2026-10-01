# ------------------------------------------------------------
# src/result_logger.py – Utility to log analysis results for history tracking
# ------------------------------------------------------------
import csv
from pathlib import Path
from datetime import datetime

# Directory where history CSV will be stored
HISTORY_DIR = Path("output/pitch_speed_v23")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = HISTORY_DIR / "results_history.csv"

def _ensure_header():
    """Create CSV with header if it does not exist yet."""
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "video",
        "handedness",  # 'left' or 'right'
        "release_speed_kmh",
        "release_speed_mph",
        "extension_m",
        "pitch_distance_m",
        "flight_time_s",
    ])

def log_result(
    video: str,
    is_left_handed: bool,
    release_speed_kmh: float,
    extension_m: float,
    pitch_distance_m: float,
    flight_time_s: float,
):
    """Append a single analysis result to the history CSV.

    Parameters
    ----------
    video : str
        動画名（例: "pitch_102_1"）
    is_left_handed : bool
        左投げなら True, 右投げなら False
    release_speed_kmh : float
        初速（km/h）
    extension_m : float
        エクステンション（メートル）
    pitch_distance_m : float
        投球距離（メートル）
    flight_time_s : float
        飛行時間（秒）
    """
    # Compute MPH from km/h
    release_speed_mph = release_speed_kmh * 0.621371
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            video,
            handedness,
            f"{release_speed_kmh:.2f}",
            f"{release_speed_mph:.2f}",
            f"{extension_m:.2f}",
            f"{pitch_distance_m:.2f}",
            f"{flight_time_s:.4f}",
        ])
