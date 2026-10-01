from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 設定
# ============================================================

FILES = [
    Path("output/my_3pitches_test/pitch_1_tracking.csv"),
    Path("output/my_3pitches_test/pitch_2_tracking.csv"),
    Path("output/my_3pitches_test/pitch_3_tracking.csv"),
]

# ============================================================
# 開始
# ============================================================

print("=" * 60)
print("自分の3投球 追跡結果比較")
print("=" * 60)

dfs = []

for i, file in enumerate(FILES, start=1):

    print()
    print("-" * 60)
    print(f"投球 {i}")
    print("-" * 60)
    print("ファイル:", file)

    if not file.exists():
        print("ERROR: ファイルがありません")
        continue

    df = pd.read_csv(file)

    dfs.append(df)

    print("行数:", len(df))
    print("列:", list(df.columns))

    print()
    print("先頭データ:")
    print(df.head(20).to_string(index=False))

    print()
    print("座標統計:")
    print(f"center_x 平均 : {df['center_x'].mean():.6f}")
    print(f"center_x 最小 : {df['center_x'].min():.6f}")
    print(f"center_x 最大 : {df['center_x'].max():.6f}")

    print(f"center_y 平均 : {df['center_y'].mean():.6f}")
    print(f"center_y 最小 : {df['center_y'].min():.6f}")
    print(f"center_y 最大 : {df['center_y'].max():.6f}")

    print(f"confidence 平均 : {df['confidence'].mean():.6f}")
    print(f"confidence 最小 : {df['confidence'].min():.6f}")
    print(f"confidence 最大 : {df['confidence'].max():.6f}")


# ============================================================
# 3つのCSVが完全一致しているか
# ============================================================

print()
print("=" * 60)
print("3本の追跡CSV比較")
print("=" * 60)

if len(dfs) == 3:

    df1 = dfs[0]
    df2 = dfs[1]
    df3 = dfs[2]

    same_12 = df1.equals(df2)
    same_13 = df1.equals(df3)
    same_23 = df2.equals(df3)

    print()
    print("完全一致:")
    print("投球1 vs 投球2:", same_12)
    print("投球1 vs 投球3:", same_13)
    print("投球2 vs 投球3:", same_23)

    # --------------------------------------------------------
    # 数値の最大差
    # --------------------------------------------------------

    common_rows = min(len(df1), len(df2), len(df3))

    print()
    print("最大差:")

    for column in ["frame", "time_sec", "center_x", "center_y", "confidence"]:

        if column not in df1.columns:
            continue

        a = pd.to_numeric(df1[column].iloc[:common_rows], errors="coerce")
        b = pd.to_numeric(df2[column].iloc[:common_rows], errors="coerce")
        c = pd.to_numeric(df3[column].iloc[:common_rows], errors="coerce")

        diff12 = np.nanmax(np.abs(a.values - b.values))
        diff13 = np.nanmax(np.abs(a.values - c.values))
        diff23 = np.nanmax(np.abs(b.values - c.values))

        print()
        print(column)
        print(f"  1 vs 2 : {diff12:.10f}")
        print(f"  1 vs 3 : {diff13:.10f}")
        print(f"  2 vs 3 : {diff23:.10f}")

    # --------------------------------------------------------
    # 座標だけ比較
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("座標比較")
    print("=" * 60)

    print()

    for i in range(common_rows):

        x1 = df1.iloc[i]["center_x"]
        y1 = df1.iloc[i]["center_y"]

        x2 = df2.iloc[i]["center_x"]
        y2 = df2.iloc[i]["center_y"]

        x3 = df3.iloc[i]["center_x"]
        y3 = df3.iloc[i]["center_y"]

        print(
            f"row {i:02d} | "
            f"1=({x1:.1f},{y1:.1f}) "
            f"2=({x2:.1f},{y2:.1f}) "
            f"3=({x3:.1f},{y3:.1f})"
        )

else:

    print()
    print("3本すべてのCSVを読み込めませんでした。")


# ============================================================
# 完了
# ============================================================

print()
print("=" * 60)
print("比較完了")
print("=" * 60)
