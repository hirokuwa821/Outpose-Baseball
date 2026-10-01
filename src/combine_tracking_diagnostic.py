import pandas as pd

DIAGNOSTIC = "output/ball_model_diagnostic.csv"
QUALITY = "output/ball_tracking_quality.csv"
OUTPUT = "output/ball_tracking_diagnostic_combined.csv"

diagnostic = pd.read_csv(DIAGNOSTIC)
quality = pd.read_csv(QUALITY)

# 拡張子やCSV名の違いを吸収して結合用キーを作る
diagnostic["video_key"] = diagnostic["video"].str.replace(".mp4", "", regex=False)
quality["video_key"] = quality["video"].str.replace("_tracking.csv", "", regex=False)

# 必要な追跡品質情報だけ結合
quality_cols = [
    "video_key",
    "frames",
    "start_frame",
    "end_frame",
    "mean_conf",
    "min_conf",
    "straightness",
]

combined = diagnostic.merge(quality[quality_cols], on="video_key", how="left")

# 絶対誤差
if "error_kmh" in combined.columns:
    combined["abs_error"] = combined["error_kmh"].abs()
elif "error" in combined.columns:
    combined["abs_error"] = combined["error"].abs()
elif "prediction_error" in combined.columns:
    combined["abs_error"] = combined["prediction_error"].abs()
else:
    combined["abs_error"] = (combined["speed_kmh"] - combined["predicted_kmh"]).abs()

# 誤差の大きい順
combined = combined.sort_values("abs_error", ascending=False)

# 保存
combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print()
print("=== 予測誤差 × 追跡品質 ===")

columns = [
    "video",
    "speed_kmh",
    "predicted_kmh",
    "abs_error",
    "frames",
    "mean_conf",
    "straightness",
]

# 実際に存在する列だけ使用
columns = [c for c in columns if c in combined.columns]

print(combined[columns].head(15).to_string(index=False))

print()
print(f"結合動画数: {len(combined)}")
print(f"結果保存: {OUTPUT}")
