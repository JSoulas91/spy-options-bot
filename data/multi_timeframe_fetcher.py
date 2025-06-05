# data/multi_timeframe_fetcher.py
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any

from utils.logger import bot_logger

TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

# ────────────────────────────────────────────────────────────
# 💾  SIMPLE IN‑MEMORY CACHES
# ────────────────────────────────────────────────────────────
_INTRADAY_CACHE: Dict[str, Dict[str, Any]] = {}
_HISTORY_CACHE: Dict[str, Dict[str, Any]] = {}

INTRADAY_TTL_SEC = 60          # refresh 1‑min, 5‑min … once per minute
HISTORY_TTL_HRS  = 12          # refresh 5d‑6mo daily history twice per day


# ────────────────────────────────────────────────────────────
# 📈  INDICATOR HELPERS
# ────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return {}

    df = df.copy()

    # EMA 20, 50, 200
    df["ema_20"] = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI (14)
    df["rsi"] = compute_rsi(df["close"], period=14)

    # MACD & Signal
    df["macd"], df["macd_signal"] = compute_macd(df["close"])

    # ATR (14)
    df["atr"] = compute_atr(df, period=14)

    # VWAP
    df["vwap"] = compute_vwap(df)

    # Bollinger Bands (20, 2 std)
    ma20 = df["close"].rolling(window=20)
    df["bb_upper"] = ma20.mean() + 2 * ma20.std()
    df["bb_lower"] = ma20.mean() - 2 * ma20.std()

    # Support / Resistance (14‑bar pivot high/low)
    support, resistance = compute_support_resistance(df)

    last = df.iloc[-1]
    return {
        "price":   last["close"],
        "ema_20":  last["ema_20"],
        "ema_50":  last["ema_50"],
        "ema_200": last["ema_200"],
        "rsi":     last["rsi"],
        "macd":          last["macd"],
        "macd_signal":   last["macd_signal"],
        "atr":     last["atr"],
        "vwap":    last["vwap"],
        "bb_upper": last["bb_upper"],
        "bb_lower": last["bb_lower"],
        "support":  support,
        "resistance": resistance,
    }

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
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_vwap(df):
    return (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

def compute_support_resistance(df):
    pivots = df["close"].rolling(14)
    support    = pivots.min().iloc[-1] if not pivots.min().empty else None
    resistance = pivots.max().iloc[-1] if not pivots.max().empty else None
    return support, resistance


# ────────────────────────────────────────────────────────────
# 🌐  RAW DATA FETCHERS  (wrapped with cache)
# ────────────────────────────────────────────────────────────
def _cached_fetch(
    cache: dict,
    key: str,
    ttl_seconds: int,
    fetch_fn,
    *args, **kwargs
):
    now = datetime.utcnow()
    entry = cache.get(key, {})
    if entry and now < entry["expires"]:
        return entry["data"]

    data = fetch_fn(*args, **kwargs)
    if data is not None:
        cache[key] = {"data": data, "expires": now + timedelta(seconds=ttl_seconds)}
    return data

def _fetch_timesales(symbol, start_dt, end_dt, interval):
    url = f"{TRADIER_BASE_URL}/markets/timesales"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "end":   end_dt.strftime("%Y-%m-%dT%H:%M"),
        "session_filter": "open"
    }
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code != 200:
        bot_logger.warning(f"[Tradier Timesales {interval}] {r.status_code}")
        return None
    j = r.json()
    if "series" not in j or "data" not in j["series"]:
        bot_logger.warning(f"[Tradier Timesales {interval}] empty")
        return None
    df = pd.DataFrame(j["series"]["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

def _fetch_history(symbol, start_date, end_date):
    url = f"{TRADIER_BASE_URL}/markets/history"
    params = {
        "symbol": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end":   end_date.strftime("%Y-%m-%d"),
        "interval": "daily"
    }
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code != 200:
        bot_logger.warning(f"[Tradier History] {symbol} {r.status_code}")
        return None
    j = r.json()
    if "history" not in j or j["history"] is None:
        bot_logger.warning(f"[Tradier History] no data {symbol}")
        return None
    rows = j["history"].get("day", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ────────────────────────────────────────────────────────────
# 🏗️  MAIN PUBLIC FUNCTION
# ────────────────────────────────────────────────────────────
def get_multi_timeframe_data(symbol: str = "SPY"):
    """
    Returns a dict:
        {
          "daily": { "5d": {...}, ... },
          "intraday": { "1min_5d": {...}, ... },
          "merged": { **all_features }
        }
    with automatic caching to limit Tradier API usage.
    """
    now = datetime.utcnow()

    # ---------------- Long‑term daily ranges ----------------
    lookbacks = {
        "5d":  5,
        "10d": 10,
        "15d": 15,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
    }
    daily_features = {}
    for label, days in lookbacks.items():
        start = (now - timedelta(days=days)).date()
        cache_key = f"{symbol}_hist_{label}"
        df = _cached_fetch(
            _HISTORY_CACHE,
            cache_key,
            ttl_seconds=HISTORY_TTL_HRS * 3600,
            fetch_fn=_fetch_history,
            symbol=symbol,
            start_date=start,
            end_date=now.date(),
        )
        daily_features[label] = compute_indicators(df)

    # ---------------- Intraday ranges ----------------
    intraday_cfg = {
        "1min_5d":   ("1min",  5),
        "5min_5d":   ("5min",  5),
        "15min_15d": ("15min", 15),
        "1hr_30d":   ("1hour", 30),
        "1d_6mo":    ("daily", 180),
    }
    intraday_features = {}
    for label, (interval, days) in intraday_cfg.items():
        start_dt = now - timedelta(days=days)
        cache_key = f"{symbol}_intra_{label}"
        df = _cached_fetch(
            _INTRADAY_CACHE,
            cache_key,
            ttl_seconds=INTRADAY_TTL_SEC,
            fetch_fn=_fetch_timesales,
            symbol=symbol,
            start_dt=start_dt,
            end_dt=now,
            interval=interval,
        )
        intraday_features[label] = compute_indicators(df)

    merged = {**daily_features, **intraday_features}
    return {
        "daily": daily_features,
        "intraday": intraday_features,
        "merged": merged,
    }

# ◀︎ compatibility alias (strategy imports this name)
fetch_long_term_features = get_multi_timeframe_data