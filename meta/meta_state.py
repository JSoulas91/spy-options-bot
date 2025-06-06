# meta/meta_state.py
"""
Meta‑state builders for entry & exit.
• Dynamic feature normalization (auto‑scales to recent history, falls back to defaults)
• Live SPY & option quote integration for exit state
• Lightweight 6‑second TTL caches for quotes to stay far below 60 calls/min
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import lru_cache
from typing import Dict, Tuple, List

import numpy as np
import pytz

from utils.logger import bot_logger as logger
from utils.vix_utils import get_vix_level
from data.multi_timeframe_fetcher import get_minutes_since_open, get_spy_latest_quote
from data.options_fetcher import get_quote as get_option_quote

eastern = pytz.timezone("US/Eastern")

# ────────────────────────────────────────────────────────────
# Static fallback ranges (used if dynamic calc unavailable)
# ────────────────────────────────────────────────────────────
DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "RSI": (0, 100),
    "MACD": (-5, 5),
    "EMA_DIST": (-10, 10),
    "VOL": (0, 10_000_000),
    "CONF": (0, 1),
    "DURATION": (0, 390),   # minutes
    "PROFIT": (-1, 1),
    "VIX": (10, 40),
    "SPY_ABS": (350, 500),
    "IV": (0, 1),
    "DELTA": (-1, 1),
}

# ────────────────────────────────────────────────────────────
# Normalization helper (clips to [0,1] for safety)
# ────────────────────────────────────────────────────────────
def normalize(value: float, r: Tuple[float, float]) -> float:
    lo, hi = r
    if hi - lo == 0:
        return 0.5
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))

# ────────────────────────────────────────────────────────────
# Dynamic range cache
# ────────────────────────────────────────────────────────────
_DYNAMIC_CACHE: Dict[str, Tuple[Tuple[float, float], float]] = {}  # {feat: (range, timestamp)}
_DYNAMIC_TTL = 3600  # recompute every hour


def _calc_dynamic_range(feature: str, long_term_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Scan long_term_data DataFrames for feature values and return (min,max).
    """
    vals: List[float] = []
    for df in long_term_data.values():
        if df is None or df.empty:
            continue
        if feature == "EMA_DIST":
            vals.extend((df["price"] - df["ema_20"]).tolist())
        else:
            vals.extend(df.get(feature, []).tolist())
    if not vals:
        return DEFAULT_RANGES[feature]
    return min(vals), max(vals)


def get_range(feature: str, long_term_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Return dynamic range if cached & fresh, else compute and cache.
    """
    now = time.time()
    cached = _DYNAMIC_CACHE.get(feature)
    if cached and now - cached[1] < _DYNAMIC_TTL:
        return cached[0]

    rng = _calc_dynamic_range(feature, long_term_data)
    # avoid degenerate 0 width
    if rng[0] == rng[1]:
        rng = DEFAULT_RANGES[feature]
    _DYNAMIC_CACHE[feature] = (rng, now)
    return rng


# ────────────────────────────────────────────────────────────
# Past‑trade summarizer (mean profit & duration)
# ────────────────────────────────────────────────────────────
def summarize_past_trades(trades, rng_profit, rng_duration):
    if not trades:
        return [0.5, 0.5]
    profits   = [t.get("profit", 0) for t in trades]
    durations = [t.get("duration", 0) for t in trades]
    return [
        normalize(np.mean(profits), rng_profit),
        normalize(np.mean(durations), rng_duration),
    ]


# ────────────────────────────────────────────────────────────
# Entry meta‑state builder
# ────────────────────────────────────────────────────────────
def build_meta_state_for_entry(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    confidence_score: float,
    trade_type: int,
    past_trades=None,
    long_term_data=None,
) -> np.ndarray:
    past_trades   = past_trades or []
    long_term_data = long_term_data or {}

    try:
        # Dynamic ranges
        rsi_rng   = get_range("RSI", long_term_data)
        macd_rng  = get_range("MACD", long_term_data)
        ema_rng   = get_range("EMA_DIST", long_term_data)
        vol_rng   = get_range("VOL", long_term_data)
        dur_rng   = DEFAULT_RANGES["DURATION"]
        profit_rng= DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            last = df.iloc[-1]
            return [
                normalize(last.get("rsi", 50), rsi_rng),
                normalize(last.get("macd", 0), macd_rng),
                normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                normalize(last.get("volume", 0), vol_rng),
            ]

        state = [
            normalize(confidence_score, DEFAULT_RANGES["CONF"]),
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            normalize(get_vix_level(), DEFAULT_RANGES["VIX"]),
            *summarize_past_trades(past_trades, profit_rng, dur_rng),
            *tf_feats(data_1m),
            *tf_feats(data_5m),
            *tf_feats(data_15m),
            *tf_feats(data_1h),
            *tf_feats(data_1d),
        ]

        # Long‑term trends
        for p in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            df = long_term_data.get(p)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                state += [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                ]
            else:
                state += [0.5, 0.5, 0.5]

        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"[MetaState] entry build error: {e}")
        return np.zeros(64, dtype=np.float32)  # keep consistent length


# ────────────────────────────────────────────────────────────
# Cached option quote helper (TTL 6 s)
# ────────────────────────────────────────────────────────────
_OPTION_CACHE: Dict[str, Tuple[dict, float]] = {}
def _cached_option_quote(symbol: str, ttl=6):
    now = time.time()
    entry = _OPTION_CACHE.get(symbol)
    if entry and now - entry[1] < ttl:
        return entry[0]
    q = get_option_quote(symbol)
    _OPTION_CACHE[symbol] = (q, now)
    return q


# ────────────────────────────────────────────────────────────
# Exit meta‑state builder (live SPY & option quote)
# ────────────────────────────────────────────────────────────
def build_meta_state_for_exit(
    trade: dict,
    past_trades=None,
    long_term_data=None,
) -> np.ndarray:

    past_trades   = past_trades or []
    long_term_data = long_term_data or {}

    try:
        # Pull live SPY (cached 6s in fetcher)
        spy_q     = get_spy_latest_quote() or {}
        spy_price = spy_q.get("price", 0)

        # Pull live option quote (cached 6s / contract)
        opt_sym   = trade.get("option_symbol")
        opt_q     = _cached_option_quote(opt_sym) if opt_sym else {}
        iv        = float(opt_q.get("iv", 0) or 0)
        delta     = float(opt_q.get("delta", 0) or 0)

        # Dynamic ranges
        dur_rng   = DEFAULT_RANGES["DURATION"]
        profit_rng= DEFAULT_RANGES["PROFIT"]
        spy_abs_rng = DEFAULT_RANGES["SPY_ABS"]

        # Contextual
        confidence = trade.get("confidence", 0.5)
        trade_type = trade.get("trade_type", 0)
        entry_price= trade.get("entry_price", 0)
        pnl_pct    = (spy_price - entry_price) / max(entry_price, 1e-9)

        minutes_open = normalize(
            (datetime.now(eastern) -
             datetime.fromisoformat(trade.get("timestamp")).astimezone(eastern)
            ).seconds // 60,
            dur_rng
        )

        state = [
            normalize(confidence, DEFAULT_RANGES["CONF"]),
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            minutes_open,
            normalize(pnl_pct, profit_rng),
            normalize(get_vix_level(), DEFAULT_RANGES["VIX"]),
            *summarize_past_trades(past_trades, profit_rng, dur_rng),
            normalize(spy_price - entry_price, DEFAULT_RANGES["EMA_DIST"]),
            normalize(spy_price, spy_abs_rng),
            normalize(iv, DEFAULT_RANGES["IV"]),
            normalize(delta, DEFAULT_RANGES["DELTA"]),
        ]
        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"[MetaState] exit build error: {e}")
        return np.zeros(32, dtype=np.float32)