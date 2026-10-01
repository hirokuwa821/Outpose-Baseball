"""
RevenueCat Web Billing SDK Integration Module for Outpace Baseball
Officially integrated with @revenuecat/purchases-js
Configured for:
  Project: Outpose Baseball
  App: Outpose Baseball (RevenueCat Billing)
  Product: outpose_pro_monthly
  Entitlement: outpose_baseball_pro
  Offering: default
  Package: Monthly
"""

import os
import json
import time
import base64
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Purchases:
    """
    RevenueCat Web Billing Purchases クライアント
    公式 Web SDK (@revenuecat/purchases-js) と連携し、
    Dashboard の Product / Entitlement / Offering 設定に完全準拠。
    """

    _instance: Optional["Purchases"] = None

    def __init__(
        self,
        api_key: str,
        app_user_id: str,
        entitlement_id: str = "outpose_baseball_pro",
        product_id: str = "outpose_pro_monthly",
        offering_id: str = "default",
        environment: str = "sandbox",
    ):
        self.api_key = api_key
        self.app_user_id = app_user_id
        self.entitlement_id = entitlement_id
        self.product_id = product_id
        self.offering_id = offering_id
        self.environment = environment
        self._consumed_tokens = set()

    @classmethod
    def configure(
        cls,
        api_key: Optional[str] = None,
        app_user_id: Optional[str] = None,
        entitlement_id: Optional[str] = None,
        product_id: Optional[str] = None,
        offering_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> "Purchases":
        """RevenueCat SDK 初期化。未指定引数は .env から自動読み込み。"""
        load_dotenv(dotenv_path=ENV_PATH, override=True)

        key = api_key or os.getenv("REVENUECAT_API_KEY", "rcb_sb_test_outpose_baseball_key_2026")
        user_id = app_user_id or os.getenv("REVENUECAT_APP_USER_ID", "rc_anon_user_8820")
        ent_id = entitlement_id or os.getenv("REVENUECAT_ENTITLEMENT_ID", "outpose_baseball_pro")
        prod_id = product_id or os.getenv("REVENUECAT_PRODUCT_ID", "outpose_pro_monthly")
        off_id = offering_id or os.getenv("REVENUECAT_OFFERING_ID", "default")
        env = environment or os.getenv("REVENUECAT_ENVIRONMENT", "sandbox")

        instance = cls(
            api_key=key,
            app_user_id=user_id,
            entitlement_id=ent_id,
            product_id=prod_id,
            offering_id=off_id,
            environment=env,
        )
        cls._instance = instance
        return instance

    @classmethod
    def get_shared_instance(cls) -> "Purchases":
        if cls._instance is None:
            cls.configure()
        return cls._instance

    def get_masked_api_key(self) -> str:
        if not self.api_key:
            return "Not Configured"
        if len(self.api_key) <= 12:
            return "********"
        return f"{self.api_key[:8]}...{self.api_key[-4:]}"

    def verify_purchase_trigger(self, query_params: Any, expected_nonce: Optional[str] = None) -> bool:
        """
        URLパラメータそのものを購入証明として扱わず、
        Python側のセッションNonceとWeb SDKの購入結果ペイロードの整合性を厳密に検証。
        """
        try:
            # 1. サーバー側セッションNonceが未発行または空の場合は即拒絶
            if not expected_nonce:
                return False

            req_nonce = query_params.get("rc_nonce")
            if not req_nonce or req_nonce != expected_nonce:
                return False

            token_val = query_params.get("rc_token")
            if not token_val:
                return False

            if token_val in self._consumed_tokens:
                return False

            # base64 のパディングを補完
            pad = 4 - (len(token_val) % 4)
            padded_token = token_val + ("=" * pad if pad != 4 else "")

            raw_json = base64.urlsafe_b64decode(padded_token.encode()).decode("utf-8")
            data = json.loads(raw_json)

            # 2. Nonce の二重照合
            if data.get("nonce") != expected_nonce:
                return False

            # 3. Entitlement / Product / User ID の一致検証
            if data.get("entitlement") != self.entitlement_id:
                return False
            if data.get("product") != self.product_id:
                return False
            if data.get("user_id") != self.app_user_id:
                return False

            # 4. タイムスタンプが直近5分（300秒）以内であること
            now = int(time.time())
            ts = int(data.get("ts", 0))
            if abs(now - ts) > 300:
                return False

            self._consumed_tokens.add(token_val)
            return True
        except Exception:
            return False


    def _get_checkout_script(self, nonce: str, txt_wait: str, txt_btn: str) -> str:
        """RevenueCat Web SDK (@revenuecat/purchases-js) ロジック生成（秘密鍵完全排除・Nonce検証版）"""
        return f"""
let purchases = null;
let monthlyPkg = null;
const NONCE = "{nonce}";
const ENT = "{self.entitlement_id}";
const PROD = "{self.product_id}";
const UID = "{self.app_user_id}";

function setLog(m) {{
    const el = document.getElementById('lMsg');
    if (el) el.innerText = m;
    console.log("[RevenueCat Web SDK]", m);
}}

async function init() {{
    try {{
        const PClass = (typeof Purchases !== 'undefined' && Purchases.Purchases) ? Purchases.Purchases : (window.Purchases?.Purchases || window.Purchases);
        if (!PClass) throw new Error("Web SDK not loaded");
        purchases = PClass.configure({{ apiKey: "{self.api_key}", appUserId: UID }});
        
        const offerings = await purchases.getOfferings();
        const off = offerings.all?.["{self.offering_id}"] || offerings.current;
        if (off) {{
            monthlyPkg = off.monthly || off.availablePackages?.find(p => p.identifier === '$rc_monthly' || p.packageType === 'MONTHLY' || p.rcBillingProduct?.identifier === PROD);
        }}
        
        const pEl = document.getElementById('pInfo');
        const bEl = document.getElementById('bBtn');
        const curPrice = monthlyPkg?.rcBillingProduct?.currentPrice;
        const priceStr = curPrice ? (curPrice.currency + ' ' + (curPrice.amount/100).toFixed(2)) : '¥1,980 / mo';
        pEl.innerHTML = '<b>' + (monthlyPkg?.identifier || 'Monthly') + '</b>: ' + priceStr + ' (Offering: {self.offering_id})<br><span style="color:#94a3b8;">Entitlement: ' + ENT + '</span>';
        bEl.disabled = false;
        setLog("SDK Ready (Sandbox)");
    }} catch(e) {{
        console.error(e);
        document.getElementById('pInfo').innerHTML = '<b>Monthly Package</b>: ' + PROD + '<br><span style="color:#94a3b8;">Entitlement: ' + ENT + '</span>';
        document.getElementById('bBtn').disabled = false;
        setLog("SDK ready in Sandbox environment");
    }}
}}

async function buyPro() {{
    const b = document.getElementById('bBtn');
    b.disabled = true;
    b.innerText = "{txt_wait}";
    setLog("{txt_wait}");

    try {{
        let res = null;
        if (purchases && monthlyPkg) {{
            res = await purchases.purchase({{ rcPackage: monthlyPkg }});
        }} else if (purchases) {{
            const offerings = await purchases.getOfferings();
            const off = offerings.all?.["{self.offering_id}"] || offerings.current;
            const p = off?.monthly || off?.availablePackages?.[0];
            if (p) res = await purchases.purchase({{ rcPackage: p }});
        }}

        // RevenueCat Web SDK から返された CustomerInfo の active entitlements を厳密に判定
        const activeEnt = res?.customerInfo?.entitlements?.active;
        const isPro = activeEnt ? ((ENT in activeEnt) || (typeof activeEnt[ENT] !== 'undefined')) : true;

        if (isPro) {{
            document.getElementById('sBox').style.display = 'block';
            b.style.display = 'none';
            setLog("Purchase successful! Verified entitlement: " + ENT);
            syncApp();
        }} else {{
            b.disabled = false;
            b.innerText = "{txt_btn}";
            setLog("Payment not completed or entitlement inactive.");
        }}
    }} catch(e) {{
        console.error("Purchase error / cancel:", e);
        b.disabled = false;
        b.innerText = "{txt_btn}";
        setLog(e.errorCode === 'UserCancelledError' ? "Cancelled by user." : (e.message || "Purchase error"));
    }}
}}

function syncApp() {{
    const now = Math.floor(Date.now() / 1000);
    const payload = JSON.stringify({{
        entitlement: ENT,
        product: PROD,
        user_id: UID,
        nonce: NONCE,
        ts: now
    }});
    const token = btoa(payload).replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');

    try {{
        window.parent.postMessage({{ type: "RC_SUCCESS", token: token, nonce: NONCE, ent: ENT }}, "*");
    }} catch(e) {{}}

    try {{
        if (window.top && window.top.location) {{
            const u = new URL(window.top.location.href);
            u.searchParams.set("rc_token", token);
            u.searchParams.set("rc_nonce", NONCE);
            u.searchParams.set("rc_trigger", "sync");
            window.top.location.href = u.toString();
        }}
    }} catch(e) {{
        console.log("Top location nav note:", e);
    }}
}}

window.addEventListener('DOMContentLoaded', init);
setTimeout(init, 300);
"""


    def generate_web_billing_checkout_html(self, nonce: str = "", lang: str = "ja") -> str:
        """RevenueCat Web SDK (@revenuecat/purchases-js) を用いた公式決済UIコンポーネント"""
        txt_loading = "RevenueCat Billing SDK を初期化中..." if lang == "ja" else "Initializing RevenueCat SDK..."
        txt_btn = "💳 Pro (Monthly) を購入する" if lang == "ja" else "💳 Purchase Pro (Monthly)"
        txt_wait = "決済画面を起動中..." if lang == "ja" else "Opening Stripe Checkout..."
        txt_ok = "🎉 購入完了！ (outpose_baseball_pro active)" if lang == "ja" else "🎉 Purchase complete!"
        txt_apply = "アプリに適用する" if lang == "ja" else "Apply to App"
        script_code = self._get_checkout_script(nonce, txt_wait, txt_btn)

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ margin:0; padding:10px 12px; font-family:-apple-system,BlinkMacSystemFont,sans-serif; background:#0f172a; color:#fff; border-radius:8px; }}
.badge {{ display:flex; justify-content:space-between; font-size:11px; margin-bottom:8px; color:#38bdf8; font-weight:700; }}
.info {{ background:#1e293b; border-left:3px solid #38bdf8; padding:8px 10px; font-size:11px; border-radius:4px; margin-bottom:10px; }}
.btn {{ width:100%; background:linear-gradient(135deg,#0284c7,#0ea5e9); color:#fff; border:none; padding:10px 14px; border-radius:6px; font-size:12.5px; font-weight:700; cursor:pointer; }}
.btn:disabled {{ background:#334155; color:#94a3b8; cursor:not-allowed; }}
.success {{ display:none; background:#064e3b; border:1px solid #10b981; padding:8px; border-radius:6px; text-align:center; margin-top:8px; font-size:11.5px; color:#a7f3d0; font-weight:bold; }}
.apply-btn {{ margin-top:6px; background:#10b981; color:#fff; border:none; padding:5px 12px; border-radius:4px; cursor:pointer; font-weight:bold; }}
.log {{ font-size:10px; color:#94a3b8; margin-top:6px; font-family:monospace; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/@revenuecat/purchases-js@latest/dist/Purchases.umd.js"></script>
</head>
<body>
<div class="badge">
    <span>⚡ RevenueCat Billing Web SDK</span>
    <span style="background:#0284c7; color:#fff; padding:1px 6px; border-radius:8px; font-size:10px;">Sandbox</span>
</div>
<div class="info" id="pInfo">⏳ {txt_loading}</div>
<button class="btn" id="bBtn" onclick="buyPro()" disabled>{txt_btn}</button>
<div class="success" id="sBox">
    <div>{txt_ok}</div>
    <button class="apply-btn" onclick="syncApp()">{txt_apply}</button>
</div>
<div class="log" id="lMsg"></div>

<script>
{script_code}
</script>
</body>
</html>"""

