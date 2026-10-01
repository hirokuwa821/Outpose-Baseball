# Outpace Baseball ⚾
> **Analytics & Biomechanics Dashboard**  
> AI-powered baseball pitch analytics & biomechanics platform built for coaches, scouts, and players.

---

## 📌 解決する課題 (Problem & Solution)

### 1. 従来の高額な専用計測機器への依存
- **課題**: 球速・回転数・3D弾道・ホップ量（IVB）・ピッチトンネルといった高度なピッチトラッキングを行うには、数百万円規模の高額なレーダー測定機器（TrackMan, Rapsodo等）が必要であり、アマチュアや学生、育成現場では導入が困難でした。
- **解決策**: **Outpace Baseball** は、一般的なスマートフォンや単一カメラで撮影した投球動画から、AI姿勢推定と空気力学・弾道モデリングを統合し、誰でも手軽にプロ基準の投球トラッキングとStuff+球質評価を可能にします。

### 2. 直感的な球質・フォーム理解の欠如
- **課題**: 数値データ（回転数や変化量）の羅列だけでは、選手や指導者が直感的に球質を把握できず、フォーム改善や配球設計に直結しにくい課題がありました。
- **解決策**: Three.js によるリアルタイム 3D 回転ボール（フォーシーム縫い目・握り方ガイド）、Statcast 風 3D 弾道＆ピッチトンネル、および Stuff+ 5軸レーダー評価により、直感的なビジュアルフィードバックを提供します。

---

## 💎 RevenueCat Integration (Shipaton 2026)

本プロジェクトは **Shipaton 2026** 提出用として、**RevenueCat SDK（Sandbox テスト環境）** を統合しています。ストア公開不要で、ローカルおよび審査環境で安全に課金・サブスクリプション検証が行える最小構成を実装しています。

### 1. SDK 初期化処理 (`Purchases.configure`)
- **配置**: `src/revenuecat_sdk.py`, `src/app.py`
- アプリケーション起動時に `Purchases.configure()` を実行し、クライアントインスタンスをシングルトン管理します。
- 認証情報や環境設定は、後述の `.env` から動的にロードされます。

```python
# src/app.py
import revenuecat_sdk as rc

# RevenueCat SDK の初期化 (.env より API Key を安全に読み込み)
rc_purchases = rc.Purchases.configure()
```

### 2. Paywall 画面 ＆ Sandbox テスト購入
- **配置**: `src/app.py` (`show_revenuecat_paywall()`), `src/revenuecat_sdk.py`
- アプリ上部やサイドバーのボタンからモーダル形式の Paywall（料金プラン案内）を開くことができます。
- 画面上部には審査員向けの「🧪 **RevenueCat Sandbox Test Mode**」ステータスバーを表示。
- 有料プラン（Pro / Elite）の「**🧪 テスト購入**」ボタンを押下すると、Sandbox 購入ロジック (`purchases.purchase_package(plan_id)`) がシミュレート実行されます。

### 3. 購入成功時のステート切り替え (`isSubscribed = True`)
- **状態管理**: `CustomerInfo.isSubscribed`, `st.session_state["isSubscribed"]`
- テスト購入が完了すると、直ちにエンタイトルメント情報が更新されます：
  - `isSubscribed = True` および `is_premium = True` が有効化。
  - トースト通知 (`🎉 [RevenueCat Sandbox] 購入成功！`) とサイドバーのバッジ (`✅ isSubscribed = True`) が即座に更新。
  - Free プラン選択時には安全に `isSubscribed = False` へ切り替わります。

### 4. セキュリティ配慮（シークレット管理）
- **環境変数の完全分離**: API Key、ユーザー ID、エンタイトルメント ID は `.env` にのみ記述され、`python-dotenv` 経由で実行時に読み込まれます。
- **安全な Git 管理**: `.gitignore` により `.env` や認証ファイル、メディア・大容量データはすべて Git 追跡から除外されています。
- **公開用テンプレート**: 提出用として安全なプレースホルダーのみが記載された `.env.example` を提供しています。

---

## 📥 Sample Pitch Videos (デモ・審査用サンプル動画)

Streamlit Community Cloud 上で即座に投球解析や2球比較機能を試せるよう、事前検証済みのサンプル投球動画（開発者本人の実投球動画）を配布しています。

- **[📦 GitHub Releases: sample-videos ページはこちら](https://github.com/hirokuwa821/Outpose-Baseball/releases/tag/sample-videos)**

| サンプル動画 | 球速・特徴 | サイズ | ダウンロード |
|---|---|---|---|
| **Sample Pitch 1** (`pitch_102_1.mp4`) | 直球 102 km/h（高速解析・単球デモ推奨） | 1.2 MB | [📥 ダウンロード](https://github.com/hirokuwa821/Outpose-Baseball/releases/download/sample-videos/pitch_102_1.mp4) |
| **Sample Pitch 2** (`pitch_98_1.mp4`) | 直球 98 km/h（2-Pitch Comparison 比較用） | 1.8 MB | [📥 ダウンロード](https://github.com/hirokuwa821/Outpose-Baseball/releases/download/sample-videos/pitch_98_1.mp4) |

### 審査員向けクイックテスト手順:
1. 上記リンク（または [Releases ページ](https://github.com/hirokuwa821/Outpose-Baseball/releases/tag/sample-videos)）からサンプル動画（MP4）をPCまたはスマホに保存。
2. アプリのサイドバーにある「動画アップロード」からファイルを選択し、「アップロードして解析」を押下。
3. 2本両方をアップロードすると、「2投球フォーム比較」タブで2球のリリース差・ピッチトンネル比較が可能になります。

---

## 🛠️ クイックスタート

### 1. 環境構築
```bash
# Python 仮想環境の作成と有効化
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Mac / Linux
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
pip install python-dotenv
```

### 2. 環境変数の設定
`.env.example` をコピーして `.env` を作成します：
```bash
cp .env.example .env
```
`.env` ファイルを開き、設定値を確認します（テスト用ダミーキーが初期設定されています）：
```ini
REVENUECAT_API_KEY=your_revenuecat_api_key_here
REVENUECAT_APP_USER_ID=rc_anon_user_8820
REVENUECAT_ENTITLEMENT_ID=pro_access
REVENUECAT_ENVIRONMENT=sandbox
```

### 3. アプリ起動
```bash
streamlit run src/app.py
```

---

## ⚾ 機能ハイライト (高水準概要)

- **3D Spinning Baseball**: Three.js による回転軸・マグヌス揚力方向・球種別握り方の可視化。
- **Statcast 3D Trajectory & Tunnel**: 左右投手の物理空間補正、打者判断点（7.2m手前）の判定。
- **Stuff+ 5-Axis Radar**: ホップ(Ride+)、球速(Velo+)、前到達(Ext+)、スピン効率(Eff+)、横変化(Run+)のプロ平均比偏差値評価。
- **Pitch Movement Plot**: 基準球種ゾーンと実測ボール位置の対比表示。
- **Multilingual Support**: 日・英・西・韓・繁中 5言語完全対応。

