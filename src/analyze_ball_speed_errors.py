import pandas as pd

INPUT_CSV = "output/randomforest_ball_speed_predictions_20.csv"

df = pd.read_csv(INPUT_CSV)

# 絶対誤差の大きい順
df = df.sort_values("absolute_error_kmh", ascending=False)

print("=" * 70)
print("球速予測の誤差分析")
print("=" * 70)

print()
print("【動画ごとの予測結果】")
print("-" * 70)

for _, row in df.iterrows():

    print(f"{row['video']}")

    print(f"  実測: {row['actual_speed_kmh']:.2f} km/h")

    print(f"  予測: {row['predicted_speed_kmh']:.2f} km/h")

    print(f"  誤差: {row['error_kmh']:+.2f} km/h")

    print(f"  絶対誤差: {row['absolute_error_kmh']:.2f} km/h")

    print()


# ==========================================
# 誤差の統計
# ==========================================

mae = df["absolute_error_kmh"].mean()

max_error = df["absolute_error_kmh"].max()

median_error = df["absolute_error_kmh"].median()

print("=" * 70)
print("【誤差の統計】")
print("=" * 70)

print(f"平均絶対誤差 MAE   : {mae:.2f} km/h")
print(f"中央値             : {median_error:.2f} km/h")
print(f"最大絶対誤差       : {max_error:.2f} km/h")


# ==========================================
# 誤差が大きい動画
# ==========================================

print()
print("=" * 70)
print("【誤差が10 km/h以上の動画】")
print("=" * 70)

large_error = df[df["absolute_error_kmh"] >= 10]

if len(large_error) == 0:

    print("10 km/h以上の誤差はありません。")

else:

    for _, row in large_error.iterrows():

        print(
            f"{row['video']} : "
            f"実測 {row['actual_speed_kmh']:.2f} → "
            f"予測 {row['predicted_speed_kmh']:.2f} "
            f"(誤差 {row['error_kmh']:+.2f})"
        )


# ==========================================
# 保存
# ==========================================

df.to_csv(
    "output/randomforest_ball_speed_errors_sorted.csv",
    index=False,
    encoding="utf-8-sig",
)

print()
print("保存:")
print("output/randomforest_ball_speed_errors_sorted.csv")

