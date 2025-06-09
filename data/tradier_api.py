# data/tradier_api.py
import os
import time
import requests
from typing import Any, Dict, List, Optional
from utils.logger import bot_logger as logger

# ─────────────────────────────────────────────
# ░▒▓  Environment / Globals  ▓▒░
# ─────────────────────────────────────────────
TRADIER_API_TOKEN    = os.getenv("TRADIER_API_TOKEN")
TRADIER_ACCOUNT_ID   = os.getenv("TRADIER_ACCOUNT_ID")
USE_LIVE_TRADIER     = os.getenv("USE_LIVE_TRADIER", "false").lower() == "true"

if not TRADIER_API_TOKEN or not TRADIER_ACCOUNT_ID:
    raise EnvironmentError("Missing TRADIER_API_TOKEN or TRADIER_ACCOUNT_ID in environment")

BASE_URL = "https://api.tradier.com/v1" if USE_LIVE_TRADIER else "https://sandbox.tradier.com/v1"
HEADERS  = {
    "Authorization": f"Bearer {TRADIER_API_TOKEN}",
    "Accept":        "application/json"
}

# ─────────────────────────────────────────────
# ░▒▓  HTTP helper with retry / back‑off  ▓▒░
# ─────────────────────────────────────────────
def safe_request(method: str,
                 endpoint: str,
                 params: Optional[Dict[str, Any]] = None,
                 data:   Optional[Dict[str, Any]] = None,
                 max_retries: int = 3,
                 timeout: int = 10) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(max_retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=HEADERS,
                params=params,
                data=data,
                timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error(f"Tradier API error [{endpoint}] (attempt {attempt+1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.exception("Maximum retries reached – giving up.")
    return None

# ─────────────────────────────────────────────
# ░▒▓  Quote helpers  ▓▒░
# ─────────────────────────────────────────────
def _extract_quote(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (payload or {}).get("quotes", {}).get("quote", {}) or {}

def get_equity_quote(symbol: str = "SPY") -> Dict[str, Any]:
    return _extract_quote(safe_request("GET", "markets/quotes", params={"symbols": symbol}))

def get_option_quote(option_symbol: str) -> Dict[str, Any]:
    return _extract_quote(safe_request("GET", "markets/quotes", params={"symbols": option_symbol}))

# ─────────────────────────────────────────────
# ░▒▓  Option chain  ▓▒░
# ─────────────────────────────────────────────
def _normalize_contract_list(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]

def get_option_chain(symbol: str = "SPY",
                     expiration: Optional[str] = None,
                     greeks: bool = False) -> List[Dict[str, Any]]:
    params = {"symbol": symbol,
              "expiration": expiration,
              "greeks": "true" if greeks else "false"}
    data = safe_request("GET", "markets/options/chains", params=params)
    option_obj = (data or {}).get("options", {}).get("option", [])
    return _normalize_contract_list(option_obj)

# ─────────────────────────────────────────────
# ░▒▓  Orders  ▓▒░
# ─────────────────────────────────────────────
def place_option_order(option_symbol: str,
                       quantity: int,
                       side: str,
                       order_type: str = "market",
                       duration: str = "day"
                       ) -> Dict[str, Any]:
    endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
    payload = {
        "class"        : "option",
        "symbol"       : "SPY",
        "option_symbol": option_symbol,
        "side"         : side,
        "quantity"     : quantity,
        "type"         : order_type,
        "duration"     : duration
    }
    data = safe_request("POST", endpoint, data=payload)
    return (data or {}).get("order", {}) or {}

def check_order_status(order_id: str) -> Dict[str, Any]:
    data = safe_request("GET", f"accounts/{TRADIER_ACCOUNT_ID}/orders/{order_id}")
    return (data or {}).get("order", {}) or {}

def cancel_order(order_id: str) -> Dict[str, Any]:
    data = safe_request("DELETE", f"accounts/{TRADIER_ACCOUNT_ID}/orders/{order_id}")
    return (data or {}).get("order", {}) or {}

# ─────────────────────────────────────────────
# ░▒▓  Account  ▓▒░
# ─────────────────────────────────────────────
def get_account_balances() -> Dict[str, Any]:
    data = safe_request("GET", f"accounts/{TRADIER_ACCOUNT_ID}/balances")
    return (data or {}).get("balances", {}) or {}

def get_account_positions() -> List[Dict[str, Any]]:
    positions = (safe_request("GET", f"accounts/{TRADIER_ACCOUNT_ID}/positions") or {}) \
                    .get("positions", {}) \
                    .get("position", [])
    return _normalize_contract_list(positions)