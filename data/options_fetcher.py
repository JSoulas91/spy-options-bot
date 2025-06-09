# data/options_fetcher.py
"""
Lightweight helpers around Tradier’s option‑chain + quote endpoints.
Keeps a small in‑memory TTL cache so we stay well below the 60‑call/min limit.
"""

from __future__ import annotations
import os, time
from datetime import datetime, timedelta
from functools  import lru_cache
from typing     import Dict, Any, List, Optional

import requests
from utils.logger import bot_logger as logger
from config       import TRADIER_API_TOKEN, TRADIER_BASE_URL   # ← single source‑of‑truth

# ───────────────────────────────────────────────────────────
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_TOKEN}",
    "Accept":        "application/json",
}

# ── tiny TTL cache ─────────────────────────────────────────
_CACHE: dict[str, dict[str, Any]] = {}   # key → {"val":…, "exp":…}

def _get_cache(key: str, ttl: int) -> Optional[Any]:
    itm = _CACHE.get(key)
    if itm and time.time() < itm["exp"]:
        return itm["val"]
    _CACHE.pop(key, None)
    return None

def _set_cache(key: str, val: Any, ttl: int) -> None:
    _CACHE[key] = {"val": val, "exp": time.time() + ttl}

# ───────────────────────────────────────────────────────────
# Market‑open helper
# ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_minutes_since_open() -> int:
    from datetime import timezone as _tz
    import pytz
    eastern = pytz.timezone("US/Eastern")
    now     = datetime.now(eastern)
    open_   = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, int((now - open_).total_seconds() // 60))

# ───────────────────────────────────────────────────────────
# Generic GET with short retry (helps against transient 504s)
# ───────────────────────────────────────────────────────────
def _tradier_get(endpoint: str, params: dict[str, Any], retries: int = 2) -> dict[str, Any] | None:
    url = f"{TRADIER_BASE_URL}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=8)
            if r.status_code in (401, 403):
                logger.error("Tradier auth failed (status %s) – check token or sandbox flag.", r.status_code)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("[Tradier GET] %s  (attempt %d/%d)", exc, attempt, retries)
            if attempt == retries:
                return None
            time.sleep(1.5)

# ───────────────────────────────────────────────────────────
# Quote helpers
# ───────────────────────────────────────────────────────────
def get_quote(symbol: str, ttl: int = 20) -> dict[str, Any]:
    ck = f"q:{symbol}"
    if (cached := _get_cache(ck, ttl)):
        return cached

    js  = _tradier_get("markets/quotes", {"symbols": symbol, "greeks": "true"})
    q   = (js or {}).get("quotes", {}).get("quote")
    if isinstance(q, list):           # Tradier returns a list if >1 symbol
        q = q[0] if q else {}
    if q:
        _set_cache(ck, q, ttl)
    return q or {}

# ───────────────────────────────────────────────────────────
# Option‑chain helpers
# ───────────────────────────────────────────────────────────
def _normalize(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]

def get_option_chain(symbol: str,
                     expiration: str,
                     opt_type: str,
                     ttl: int = 180) -> list[dict[str, Any]]:
    ck = f"chain:{symbol}:{expiration}:{opt_type}"
    if (cached := _get_cache(ck, ttl)):
        return cached

    js = _tradier_get(
        "markets/options/chains",
        {"symbol": symbol, "expiration": expiration, "type": opt_type, "greeks": "false"},
    )
    chain = _normalize((js or {}).get("options", {}).get("option"))
    _set_cache(ck, chain, ttl)
    return chain

# ───────────────────────────────────────────────────────────
# Small helpers to pick ATM/OTM/ITM contracts
# ───────────────────────────────────────────────────────────
def _select_moneyness_contracts(contracts: list[dict[str, Any]],
                                underlying_px: float,
                                n: int = 2) -> list[dict[str, Any]]:
    contracts = [c for c in contracts if "strike" in c]
    atm = sorted(contracts, key=lambda c: abs(c["strike"] - underlying_px))[:n]
    otm = sorted((c for c in contracts if c["strike"] > underlying_px),
                 key=lambda c: c["strike"])[:n]
    itm = sorted((c for c in contracts if c["strike"] < underlying_px),
                 key=lambda c: c["strike"], reverse=True)[:n]
    return atm + otm + itm

# ───────────────────────────────────────────────────────────
# Public: quick greeks snapshot around ATM
# ───────────────────────────────────────────────────────────
def get_option_metrics(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Pulls a few near‑moneyness contracts (call & put) for the next Friday expiry
    and returns greeks/IV/price for each side.
    """
    try:
        expiry = _next_valid_expiry()
        underlying_q = get_quote(symbol)
        price = float(underlying_q.get("last", 0))
        if price == 0:
            return {}

        out: dict[str, Any] = {}
        for opt_type in ("call", "put"):
            chain = get_option_chain(symbol, expiry, opt_type)
            for c in _select_moneyness_contracts(chain, price, n=2):
                q = get_quote(c["symbol"])
                if q and {"delta", "gamma", "theta", "vega", "iv"} <= q.keys():
                    out[opt_type] = {
                        "symbol":  c["symbol"],
                        "strike":  c["strike"],
                        "delta":   q["delta"],
                        "gamma":   q["gamma"],
                        "theta":   q["theta"],
                        "vega":    q["vega"],
                        "iv":      q["iv"],
                        "price":   q.get("last"),
                        "expiry":  expiry,
                    }
                    break
        return out
    except Exception as exc:
        logger.warning("[OptionMetrics] %s", exc)
        return {}

# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _next_valid_expiry(min_dte: int = 2) -> str:
    """Return the next Friday expiry at least `min_dte` days away."""
    today = datetime.utcnow().date()
    for i in range(1, 45):
        d = today + timedelta(days=i)
        if d.weekday() == 4 and (d - today).days >= min_dte:    # Friday
            return d.strftime("%Y-%m-%d")
    raise RuntimeError("Could not find a valid Friday expiry in the next 45 days")