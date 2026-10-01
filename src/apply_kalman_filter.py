"""
apply_kalman_filter.py

トラッキングデータ(v5)に対してカルマンフィルタを適用し、
座標のブレを除去した平滑化データ(v6_kalman)を生成する
"""

import csv
import numpy as np
from pathlib import Path

INPUT_DIR = Path("output/ball_tracking_v5")
OUTPUT_DIR = Path("output/ball_tracking_v6_kalman")
MIN_CONFIDENCE = 0.3

class KalmanFilter2D:
    def __init__(self, dt, initial_x, initial_y):
        # 状態ベクトル [x, y, vx, vy]
        self.x = np.array([[initial_x], [initial_y], [0.], [0.]])
        
        # 状態遷移モデル (等速直線運動をベースにするが、毎フレーム更新される)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 観測モデル (x, yのみ観測)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 誤差共分散行列 (初期の不確実性)
        self.P = np.eye(4) * 500.
        
        # 観測ノイズ共分散 R (トラッキングAIをどれくらい疑うか。大きいほどAIの座標を無視して予測を信じる)
        self.R = np.eye(2) * 50.
        
        # プロセスノイズ共分散 Q (ボールの動きの自由度。急な変化をどれくらい許容するか)
        self.Q = np.eye(4) * 0.1

    def predict(self, dt):
        # フレーム間の時間が一定でない場合のために dt を更新
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, zx, zy):
        Z = np.array([[zx], [zy]])
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)).dot(self.P)
        return self.x[0, 0], self.x[1, 0]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 処理する動画のリスト
    videos = ["pitch_102_1", "pitch_98_1", "pitch_98_2"]

    for video_name in videos:
        input_csv = INPUT_DIR / f"{video_name}_tracking.csv"
        if not input_csv.exists():
            print(f"スキップ: {input_csv} が見つかりません。")
            continue

        rows = []
        with open(input_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if float(r["confidence"]) >= MIN_CONFIDENCE:
                    rows.append({
                        "frame": int(r["frame"]),
                        "time_sec": float(r["time_sec"]),
                        "cx": float(r["cx"]),
                        "cy": float(r["cy"]),
                        "confidence": float(r["confidence"])
                    })
        
        rows.sort(key=lambda x: x["frame"])
        if len(rows) < 3:
            continue

        # カルマンフィルタの初期化 (最初の座標を使用)
        kf = KalmanFilter2D(dt=1/60.0, initial_x=rows[0]["cx"], initial_y=rows[0]["cy"])
        
        smoothed_rows = []
        prev_time = rows[0]["time_sec"]

        for i, row in enumerate(rows):
            dt = row["time_sec"] - prev_time
            if dt <= 0:
                dt = 1/60.0  # ゼロ割防止のデフォルト値

            # 1. 現在の位置を予測 (Predict)
            pred_x, pred_y = kf.predict(dt)

            # 2. 実際のAIの検出座標で予測を補正 (Update)
            obs_x = row["cx"]
            obs_y = row["cy"]
            smooth_x, smooth_y = kf.update(obs_x, obs_y)

            # 異常な外れ値(予測から離れすぎているもの)の判定も可能ですが、
            # 今回はカルマンフィルタの補正能力をそのまま使います。
            
            smoothed_rows.append({
                "frame": row["frame"],
                "time_sec": row["time_sec"],
                "cx": smooth_x,
                "cy": smooth_y,  # 平滑化された座標！
                "raw_cx": obs_x,
                "raw_cy": obs_y,
                "confidence": row["confidence"]
            })
            prev_time = row["time_sec"]

        # 出力
        output_csv = OUTPUT_DIR / f"{video_name}_tracking.csv"
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            fields = ["frame", "time_sec", "cx", "cy", "raw_cx", "raw_cy", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(smoothed_rows)

        print(f"{video_name}: {len(rows)}点のトラッキングデータをカルマンフィルタで平滑化しました。")

if __name__ == "__main__":
    main()

