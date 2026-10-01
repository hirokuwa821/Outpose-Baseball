# Baseball Velocity AI - Android WebView Wrapper

現在のStreamlitアプリを表示するための最小構成Androidアプリ（WebViewラッパー）です。

## プロジェクト構成
```
android/
  ├── app/
  │   ├── src/main/
  │   │   ├── AndroidManifest.xml          # INTERNET権限、ネットワーク構成
  │   │   ├── java/com/outpose/baseball/
  │   │   │   └── MainActivity.kt          # WebView設定、DOM/JS有効化、ファイルアップローダー対応
  │   │   └── res/
  │   │       ├── values/strings.xml       # ★Streamlitの接続先URLを設定する場所
  │   │       ├── values/themes.xml
  │   │       └── layout/activity_main.xml
  │   └── build.gradle.kts
  ├── build.gradle.kts
  ├── settings.gradle.kts
  ├── gradle.properties
  └── gradlew.bat
```

## Streamlit URLの変更方法
接続先のURLは以下のファイル1箇所で管理されています：
`android/app/src/main/res/values/strings.xml`

```xml
<string name="streamlit_url">https://your-streamlit-app-url.streamlit.app</string>
```

- **ローカル開発環境（PC同一LANのスマホ実機から接続）**:
  `http://<PCのローカルIP>:8501` （例: `http://192.168.1.10:8501`）
- **Android Emulatorから接続**:
  `http://10.0.2.2:8501`
- **本番クラウド環境**:
  `https://<app-name>.streamlit.app`

## APKのビルド方法
WindowsのコマンドプロンプトまたはPowerShellで `android` ディレクトリに移動し、以下を実行します：

```powershell
cd android
.\gradlew.bat assembleDebug
```

ビルド完了後、以下の場所にデバッグ用APKが生成されます：
`android/app/build/outputs/apk/debug/app-debug.apk`

## 実装の特徴
- **インターネット通信権限**: `android.permission.INTERNET`, `ACCESS_NETWORK_STATE`
- **クリアテキスト通信対応**: ローカルテスト用HTTP接続（`network_security_config.xml`）
- **WebView最適化**:
  - `javaScriptEnabled = true`
  - `domStorageEnabled = true`
  - `databaseEnabled = true`
  - `mixedContentMode = MIXED_CONTENT_ALWAYS_ALLOW`
  - `mediaPlaybackRequiresUserGesture = false` (動画再生)
- **ファイル選択機能**: Streamlit上の動画アップロードに対応する `WebChromeClient.onShowFileChooser`
- **戻るボタン対応**: WebViewの閲覧履歴があれば戻り、なければアプリ終了
- **Pull to Refresh**: `SwipeRefreshLayout` による画面引っ張り更新
- **エラーハンドリング**: ネットワーク切断時の再読み込み画面
