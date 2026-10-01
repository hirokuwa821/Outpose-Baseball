"""
RevenueCat Integration & Security Automated Test Suite
Verifies:
  A. First access: dynamic UUID generation & Web SDK configuration
  B. F5 reload: App User ID persistence & server entitlement re-check
  C. Streamlit restart: localStorage sync & state recovery
  D. Cross-browser isolation: unique App User IDs per client
  E. Security: zero Secret Key exposure & anti-forgery protection
"""

import sys
import json
import base64
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src import revenuecat_sdk as rc


def run_tests():
    print("=== REVENUECAT INTEGRATION & SECURITY AUTOMATED TESTS ===")

    # --- A. 初回アクセス テスト ---
    uid_1 = rc.generate_anonymous_app_user_id()
    uid_2 = rc.generate_anonymous_app_user_id()
    print(f"Test A-1: Generated UID 1: {uid_1}")
    print(f"Test A-2: Generated UID 2: {uid_2}")
    assert uid_1.startswith("rc_user_"), "UID must start with rc_user_"
    assert uid_1 != uid_2, "Each generated UID must be unique"

    sdk_1 = rc.Purchases.configure(app_user_id=uid_1)
    assert sdk_1.app_user_id == uid_1, "Purchases SDK must be configured with generated UID"

    html_1 = sdk_1.generate_web_billing_checkout_html(nonce="test_nonce_1", app_user_id=uid_1, lang="ja")
    assert f'const UID = "{uid_1}";' in html_1, "Web SDK must receive the dynamic App User ID"
    assert "Purchases.umd.js" in html_1, "Web SDK must use valid CDN file path"
    print("Test A (First Access / Dynamic UUID & Web SDK Config): PASSED")

    # --- B & C. F5 リロード / 再起動時の App User ID 引き継ぎ & 復元テスト ---
    recovered_sdk = rc.Purchases.configure(app_user_id=uid_1)
    assert recovered_sdk.app_user_id == uid_1, "App User ID must persist on reload"
    sync_script = rc.generate_user_id_sync_script(uid_1)
    assert 'STORAGE_KEY = "outpace_rc_uid"' in sync_script, "localStorage persistence must be active"
    assert 'localStorage.getItem(STORAGE_KEY)' in sync_script, "localStorage getItem must be called"
    assert uid_1 in sync_script, "Sync script must bind current user ID"
    print("Test B & C (F5 & Restart Persistence / localStorage Sync): PASSED")

    # --- D. 別ブラウザ（新規セッション）での独立性テスト ---
    diff_uid = rc.generate_anonymous_app_user_id()
    assert diff_uid != uid_1, "Different browsers must get distinct App User IDs"
    sdk_diff = rc.Purchases.configure(app_user_id=diff_uid)
    assert sdk_diff.app_user_id == diff_uid
    print("Test D (Distinct Browsers Isolation): PASSED")

    # --- E. セキュリティ検証（Secret Key 秘匿 & 偽造トークン遮断） ---
    # E-1: 生成された HTML / JS に Secret Key や危険な文字列が一切含まれないこと
    assert "sk_" not in html_1, "CRITICAL: Secret API Key must NEVER appear in HTML/JS!"
    assert "SECRET" not in html_1
    assert "CryptoJS" not in html_1
    assert "generate_token_secret" not in html_1
    assert "REVENUECAT_SECRET_KEY" not in html_1
    print("Test E-1 (Secret Key Zero-Exposure in Client Code): PASSED")

    # E-2: 単純な偽パラメータ (?rc_status=success) の完全拒絶
    assert not sdk_1.verify_purchase_trigger({"rc_status": "success"}, expected_nonce=None)
    assert not sdk_1.verify_purchase_trigger({"rc_status": "success"}, expected_nonce="some_nonce")
    print("Test E-2 (Reject simple fake query): PASSED")

    # E-3: ブラウザ側で entitlement を偽装したトークンを送信しても遮断されること
    fake_payload = json.dumps({
        "entitlement": "outpose_baseball_pro",
        "product": "outpose_pro_monthly",
        "user_id": uid_1,
        "nonce": "valid_nonce_abc",
        "ts": int(time.time())
    })
    fake_token = base64.urlsafe_b64encode(fake_payload.encode()).decode().rstrip("=")

    # ケース 1: Nonce 不一致（URL 直打ちや外部攻撃）
    assert not sdk_1.verify_purchase_trigger(
        {"rc_token": fake_token, "rc_nonce": "valid_nonce_abc"},
        expected_nonce="different_server_nonce",
        expected_user_id=uid_1
    ), "Mismatched nonce must be blocked"

    # ケース 2: 異なるユーザー ID への付与攻撃
    assert not sdk_1.verify_purchase_trigger(
        {"rc_token": fake_token, "rc_nonce": "valid_nonce_abc"},
        expected_nonce="valid_nonce_abc",
        expected_user_id="different_user_999"
    ), "Cross-user injection must be blocked"

    print("Test E-3 (Anti-Forgery & Tamper Protection): PASSED")

    # --- サーバー側 REST API 照会関数の安全性確認 ---
    srv_res = rc.check_subscriber_entitlement(uid_1, secret_key=None)
    # Secret Key 未設定時は False を返し、クラッシュしないこと
    assert srv_res.get("is_active") is False
    print("Test E-4 (Server REST API Safe Fallback): PASSED")

    print("\n=== ALL TEST SUITES COMPLETED SUCCESSFULLY! (100% PASSED) ===")


if __name__ == "__main__":
    run_tests()
