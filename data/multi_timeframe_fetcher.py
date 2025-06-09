import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import requests
import pytz

from utils.logger import bot_logger
from data.quote_utils import get_spy_quote  # ← live SPY quote w/ 6‑sec cache

# ── Tradier auth ───────────────────────────────────────────
TRADIER_API_TOKEN = os.getenv("TRADIER_API_TOKEN")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_TOKEN}",
    "Accept": "application/json",
}

# ── In‑memory caches (thread‑safe) ─────────────────────────
_INTRADAY_CACHE: Dict[str, Dict[str, Any]] = {}
_HISTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()

INTRADAY_TTL_SEC = 6      # intraday refetch ≤10×/min
HISTORY_TTL_HRS = 12      # daily history 2×/day

# ───────────────────────────────────────────────────────────
# Market open time helper
# ───────────────────────────────────────────────────────────
def get_minutes_since_open() -> int:
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < market_open:
        return 0
    delta = now - market_open
    return delta.seconds // 60

# ───────────────────────────────────────────────────────────
# 📈 Indicator helpers
# ───────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def compute_atr(df, period=14):
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - df["close"].shift()).abs(),
         (df["low"] - df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def compute_vwap(df):
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

def compute_support_resistance(df):
    piv = df["close"].rolling(14)
    return piv.min().iloc[-1], piv.max().iloc[-1]

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["vwap"] = compute_vwap(df)
    ma20 = df["close"].rolling(20)
    df["bb_upper"] = ma20.mean() + 2 * ma20.std()
    df["bb_lower"] = ma20.mean() - 2 * ma20.std()
    sup, res = compute_support_resistance(df)

    last = df.iloc[-1]
    return {
        "price": last["close"],
        "ema_20": last["ema_20"],
        "ema_50": last["ema_50"],
        "ema_200": last["ema_200"],
        "rsi": last["rsi"],
        "macd": last["macd"],
        "macd_signal": last["macd_signal"],
        "atr": last["atr"],
        "vwap": last["vwap"],
        "bb_upper": last["bb_upper"],
        "bb_lower": last["bb_lower"],
        "support": sup,
        "resistance": res,
    }

# ───────────────────────────────────────────────────────────
# Cached fetch helpers
# ───────────────────────────────────────────────────────────
def _cached_fetch(cache: dict, key: str, ttl: int, fn, *a, **kw):
    now = datetime.utcnow()
    with _CACHE_LOCK:
        entry = cache.get(key)
        if entry and now < entry["exp"]:
            return entry["data"]
    data = fn(*a, **kw)
    if data is not None:
        with _CACHE_LOCK:
            cache[key] = {"data": data, "exp": now + timedelta(seconds=ttl)}
    return data

# ───────────────────────────────────────────────────────────
# Raw Tradier calls
# ───────────────────────────────────────────────────────────
def _fetch_timesales(symbol, start_dt, end_dt, interval):
    url = f"{TRADIER_BASE_URL}/markets/timesales"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M"),
        "session_filter": "open",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
        data = js.get("series", {}).get("data")
        if not data:
            return None
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        bot_logger.warning(f"[Timesales] {e}")
        return None

def _fetch_history(symbol, start_date, end_date):
    url = f"{TRADIER_BASE_URL}/markets/history"
    params = {
        "symbol": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "interval": "daily",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=8)
        r.raise_for_status()
        js = r.json()
        rows = js.get("history", {}).get("day", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        bot_logger.warning(f"[History] {e}")
        return None

# ───────────────────────────────────────────────────────────
# Public fetch function (used by strategy & live runner)
# ───────────────────────────────────────────────────────────
def get_multi_timeframe_data(symbol: str = "SPY") -> Dict[str, Any]:
    now = datetime.utcnow()

    # ---- Daily look‑back ranges (cached 12h) ----
    lookbacks = {"5d": 5, "10d": 10, "15d": 15, "1mo": 30, "3mo": 90, "6mo": 180}
    daily = {}
    for label, days in lookbacks.items():
        df = _cached_fetch(
            _HISTORY_CACHE, f"{symbol}_hist_{label}",
            ttl=HISTORY_TTL_HRS * 3600,
            fn=_fetch_history,
            symbol=symbol,
            start_date=(now - timedelta(days=days)).date(),
            end_date=now.date(),
        )
        daily[label] = compute_indicators(df)

    # ---- Intraday ranges (cached 6 s) ----
    cfg = {
        "1min_5d": ("1min", 5),
        "5min_5d": ("5min", 5),
        "15min_15d": ("15min", 15),
        "1hr_30d": ("1hour", 30),
        "1d_6mo": ("daily", 180),
    }
    intra = {}
    for label, (interval, days) in cfg.items():
        df = _cached_fetch(
            _INTRADAY_CACHE, f"{symbol}_intra_{label}",
            ttl=INTRADAY_TTL_SEC,
            fn=_fetch_timesales,
            symbol=symbol,
            start_dt=now - timedelta(days=days),
            end_dt=now,
            interval=interval,
        )
        intra[label] = compute_indicators(df)

    merged = {**daily, **intra}
    merged["latest_quote"] = get_spy_quote()  # live SPY price (TTL 6 s)

    return {"daily": daily, "intraday": intra, "merged": merged}

# For backward compatibility
fetch_long_term_features = get_multi_timeframe_data

# ───────────────────────────────────────────────────────────
# Alias for external usage
# ───────────────────────────────────────────────────────────
get_spy_latest_quote = get_spy_quote  # for meta_state compatibility