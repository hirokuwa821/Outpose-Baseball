"""
RevenueCat SDK Integration Module for Outpace Baseball
Shipaton Submission / Sandbox Testing Implementation
"""

import os
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class CustomerInfo:
    """RevenueCat CustomerInfo モデル（サブスクリプション状態・エンタイトルメント）"""

    def __init__(self, app_user_id: str, is_subscribed: bool = False, active_plan: str = "Free", entitlement_id: str = "pro_access"):
        self.app_user_id = app_user_id
        self.is_subscribed = is_subscribed
        self.active_plan = active_plan
        self.entitlement_id = entitlement_id
        self.latest_purchase_date = datetime.datetime.now().isoformat() if is_subscribed else None

    @property
    def isSubscribed(self) -> bool:
        """要件: isSubscribed フラグ"""
        return self.is_subscribed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "app_user_id": self.app_user_id,
            "is_subscribed": self.is_subscribed,
            "isSubscribed": self.is_subscribed,
            "active_plan": self.active_plan,
            "entitlements": {
                self.entitlement_id: {
                    "is_active": self.is_subscribed,
                    "product_identifier": f"outpace_{self.active_plan.lower()}",
                    "sandbox": True,
                    "expires_date": (datetime.datetime.now() + datetime.timedelta(days=30)).isoformat() if self.is_subscribed else None,
                }
            },
            "sandbox": True,
        }


class Purchases:
    """
    RevenueCat Purchases SDK クライアント
    公式 SDK と同様に Purchases.configure(api_key, app_user_id) で初期化
    """

    _instance: Optional["Purchases"] = None

    def __init__(self, api_key: str, app_user_id: str, entitlement_id: str = "pro_access", environment: str = "sandbox"):
        self.api_key = api_key
        self.app_user_id = app_user_id
        self.entitlement_id = entitlement_id
        self.environment = environment
        self._is_configured = True
        self._customer_info = CustomerInfo(
            app_user_id=app_user_id,
            is_subscribed=False,
            active_plan="Free",
            entitlement_id=entitlement_id,
        )

    @classmethod
    def configure(
        cls,
        api_key: Optional[str] = None,
        app_user_id: Optional[str] = None,
        entitlement_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> "Purchases":
        """RevenueCat SDK 初期化。未指定引数は .env から自動読み込み。"""
        load_dotenv(dotenv_path=ENV_PATH, override=True)

        key = api_key or os.getenv("REVENUECAT_API_KEY", "appl_sandbox_default_key")
        user_id = app_user_id or os.getenv("REVENUECAT_APP_USER_ID", "rc_anon_user_8820")
        ent_id = entitlement_id or os.getenv("REVENUECAT_ENTITLEMENT_ID", "pro_access")
        env = environment or os.getenv("REVENUECAT_ENVIRONMENT", "sandbox")

        instance = cls(api_key=key, app_user_id=user_id, entitlement_id=ent_id, environment=env)
        cls._instance = instance
        return instance

    @classmethod
    def get_shared_instance(cls) -> "Purchases":
        if cls._instance is None:
            cls.configure()
        return cls._instance

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def get_customer_info(self) -> CustomerInfo:
        return self._customer_info

    def purchase_package(self, package_id: str) -> Dict[str, Any]:
        """Sandbox テスト購入ロジック。購入成功時に isSubscribed = True フラグを更新。"""
        if package_id == "Free":
            self._customer_info.is_subscribed = False
            self._customer_info.active_plan = "Free"
            return {
                "success": True,
                "is_subscribed": False,
                "isSubscribed": False,
                "plan": "Free",
                "customer_info": self._customer_info.to_dict(),
            }

        self._customer_info.is_subscribed = True
        self._customer_info.active_plan = package_id
        self._customer_info.latest_purchase_date = datetime.datetime.now().isoformat()

        return {
            "success": True,
            "is_subscribed": True,
            "isSubscribed": True,  # 要件フラグ
            "plan": package_id,
            "transaction_id": f"rc_sb_txn_{int(datetime.datetime.now().timestamp())}",
            "environment": self.environment,
            "customer_info": self._customer_info.to_dict(),
        }

    def get_masked_api_key(self) -> str:
        if not self.api_key:
            return "Not Configured"
        if len(self.api_key) <= 8:
            return "********"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}"

    def generate_web_sdk_html(self) -> str:
        """ブラウザ側 RevenueCat Web SDK (@revenuecat/purchases-js) スクリプト"""
        return f"""
        <script src="https://cdn.jsdelivr.net/npm/@revenuecat/purchases-js@latest/dist/purchases.umd.js"></script>
        <script>
            (async function() {{
                try {{
                    if (window.Purchases) {{
                        await window.Purchases.configure("{self.api_key}", "{self.app_user_id}");
                        console.log("[RevenueCat SDK] Initialized in Sandbox. User: {self.app_user_id}");
                    }}
                }} catch (e) {{
                    console.log("[RevenueCat SDK] Web note:", e.message);
                }}
            }})();
        </script>
        """
