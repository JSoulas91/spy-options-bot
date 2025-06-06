import os
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Any, List

import requests
from utils.logger import bot_logger

TRADIER_API_KEY   = os.getenv("TRADIER_API_KEY")
TRADIER_BASE_URL  = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept":        "application/json",
}

# ── tiny TTL cache ─────────────────────────────────────────────────
_cache: Dict[str, Any] = {}
def _get_cache(key: str, ttl: int):
    item = _cache.get(key)
    if item and time.time() < item["exp"]:
        return item["val"]
    _cache.pop(key, None)
    return None
def _set_cache(key: str, val: Any, ttl: int):
    _cache[key] = {"val": val, "exp": time.time() + ttl}

# ── Basic helpers ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_upcoming_fridays(count=3, min_dte=2) -> List[str]:
    today = datetime.now().date()
    fridays = []
    for i in range(1, 45):
        d = today + timedelta(days=i)
        if d.weekday() == 4 and (d - today).days >= min_dte:
            fridays.append(d.strftime("%Y-%m-%d"))
        if len(fridays) >= count:
            break
    return fridays

def get_quote(symbol: str, ttl: int = 20) -> Dict[str, Any]:
    ck = f"q:{symbol}"
    cached = _get_cache(ck, ttl)
    if cached:
        return cached
    try:
        r = requests.get(
            f"{TRADIER_BASE_URL}/markets/quotes",
            params={"symbols": symbol, "greeks": "true"},
            headers=HEADERS, timeout=6,
        )
        data = r.json().get("quotes", {}).get("quote")
        if isinstance(data, dict):
            _set_cache(ck, data, ttl)
            return data
    except Exception as e:
        bot_logger.warning(f"[Quote] {e}")
    return {}

def get_option_chain(symbol: str, expiry: str, opt_type: str, ttl: int = 180):
    ck = f"chain:{symbol}:{expiry}:{opt_type}"
    cached = _get_cache(ck, ttl)
    if cached:
        return cached
    try:
        r = requests.get(
            f"{TRADIER_BASE_URL}/markets/options/chains",
            params={"symbol": symbol, "expiration": expiry, "type": opt_type, "greeks": "false"},
            headers=HEADERS, timeout=6,
        )
        data = r.json().get("options", {}).get("option", [])
        if not isinstance(data, list):
            data = [data]
        _set_cache(ck, data, ttl)
        return data
    except Exception as e:
        bot_logger.warning(f"[Chain] {e}")
        return []

def select_moneyness_contracts(contracts, underlying, n=2):
    sorted_c = sorted(contracts, key=lambda c: abs(c["strike"] - underlying))
    atm = sorted_c[:n]
    otm = sorted([c for c in contracts if c["strike"] > underlying], key=lambda c: c["strike"])[:n]
    itm = sorted([c for c in contracts if c["strike"] < underlying], key=lambda c: c["strike"], reverse=True)[:n]
    return atm + otm + itm

def get_option_metrics(symbol="SPY") -> Dict[str, Any]:
    try:
        expiry = get_upcoming_fridays(1, 2)[0]
        underlying_q = get_quote(symbol)
        if not underlying_q:
            return {}
        price = float(underlying_q.get("last", 0))
        metrics = {}

        for opt_type in ("call", "put"):
            chain = get_option_chain(symbol, expiry, opt_type)
            sel   = select_moneyness_contracts(chain, price, 2)
            for c in sel:
                q = get_quote(c["symbol"])
                if q and {"delta", "gamma", "theta", "vega", "iv"} <= q.keys():
                    metrics[opt_type] = {
                        "symbol": c["symbol"],
                        "strike": c["strike"],
                        "delta":  q["delta"],
                        "gamma":  q["gamma"],
                        "theta":  q["theta"],
                        "vega":   q["vega"],
                        "iv":     q["iv"],
                        "price":  q.get("last"),
                        "expiry": expiry,
                    }
                    break
        return metrics
    except Exception as e:
        bot_logger.warning(f"[OptionMetrics] {e}")
        return {}