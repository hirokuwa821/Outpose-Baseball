from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 設定
# ============================================================

DATASET = Path("output/hand_dataset.csv")


# ============================================================
# メイン
# ============================================================


def main():

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    df = pd.read_csv(DATASET)

    print("=" * 60)
    print("利き腕分類AI 学習開始")
    print("=" * 60)

    print(f"データ数: {len(df)}")
    print(f"列数: {len(df.columns)}")

    # --------------------------------------------------------
    # ラベル
    # --------------------------------------------------------

    y = df["hand"]

    print()
    print("ラベル分布")
    print(y.value_counts())

    # --------------------------------------------------------
    # 特徴量
    # --------------------------------------------------------

    # video_name と hand は学習に使わない
    X = df.drop(columns=["video_name", "hand"])

    # 数値以外があれば除外
    X = X.select_dtypes(include=["number"])

    # 欠損値処理
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    X = X.fillna(X.median())

    print()
    print(f"使用特徴量数: {X.shape[1]}")

    # --------------------------------------------------------
    # ラベルを数値化
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    print()
    print("クラス")
    print(dict(zip(encoder.classes_, range(len(encoder.classes_)))))

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Stratified K-Fold
    # --------------------------------------------------------

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print()
    print("=" * 60)
    print("5-fold Cross Validation")
    print("=" * 60)

    predictions = cross_val_predict(model, X, y_encoded, cv=cv, n_jobs=-1)

    # --------------------------------------------------------
    # 評価
    # --------------------------------------------------------

    accuracy = accuracy_score(y_encoded, predictions)

    print()
    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("Classification Report")
    print(
        classification_report(
            y_encoded, predictions, target_names=encoder.classes_, digits=4
        )
    )

    # --------------------------------------------------------
    # 混同行列
    # --------------------------------------------------------

    cm = confusion_matrix(y_encoded, predictions)

    print()
    print("Confusion Matrix")
    print(cm)

    print()
    print("行 = 実際のラベル / 列 = AIの予測")

    print(f"実際 right → 予測 right: {cm[0, 0]}")

    print(f"実際 right → 予測 left : {cm[0, 1]}")

    print(f"実際 left  → 予測 right: {cm[1, 0]}")

    print(f"実際 left  → 予測 left : {cm[1, 1]}")

    # --------------------------------------------------------
    # 全データで最終モデルを学習
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("最終モデル学習")
    print("=" * 60)

    model.fit(X, y_encoded)

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    )

    importance = importance.sort_values("importance", ascending=False)

    print()
    print("=" * 60)
    print("重要特徴量 TOP 20")
    print("=" * 60)

    print(importance.head(20).to_string(index=False))

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    output = Path("output/hand_feature_importance.csv")

    importance.to_csv(output, index=False)

    print()
    print(f"重要特徴量を保存: {output}")

    print()
    print("=" * 60)
    print("利き腕分類AI 学習完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
