"""
Meta‑state builders (entry / exit)
──────────────────────────────────
• Dynamic feature normalisation (auto‑scales to LT history)
• Position‑size awareness
• Market‑regime one‑hot encoding
• Live SPY & option quote for exit state
• 6‑second TTL caches to stay far below 60 Tradier calls / min
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pytz

from utils.logger import bot_logger as logger
from utils.vix_utils import fetch_vix_price          # ← updated import
from data.multi_timeframe_fetcher import (
    get_minutes_since_open,
    get_spy_latest_quote,
)
from data.options_fetcher import get_quote as get_option_quote
from config import MAX_POSITION_SIZE  # position‑sizing awareness

# ───────────────────────────────────────────────
eastern = pytz.timezone("US/Eastern")

DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "RSI": (0, 100),
    "MACD": (-5, 5),
    "EMA_DIST": (-10, 10),
    "VOL": (0, 10_000_000),
    "CONF": (0, 1),
    "DURATION": (0, 390),
    "PROFIT": (-1, 1),
    "VIX": (10, 40),
    "SPY_ABS": (350, 500),
    "IV": (0, 1),
    "DELTA": (-1, 1),
    "SIZE": (0, MAX_POSITION_SIZE),
}

def normalize(val: float, rng: Tuple[float, float]) -> float:
    lo, hi = rng
    return 0.5 if hi - lo == 0 else float(max(0, min(1, (val - lo) / (hi - lo))))

# ───────────────────────────────────────────────
# Dynamic range cache
_DYNAMIC: Dict[str, Tuple[Tuple[float, float], float]] = {}
_DYN_TTL = 3600  # 1 h

def _calc_range(feat: str, long_term: Dict[str, np.ndarray]) -> Tuple[float, float]:
    vals: List[float] = []
    for df in long_term.values():
        if df is None or df.empty:
            continue
        if feat == "EMA_DIST":
            vals.extend((df["price"] - df["ema_20"]).tolist())
        else:
            vals.extend(df.get(feat, []).tolist())
    return (min(vals), max(vals)) if vals else DEFAULT_RANGES[feat]

def get_range(feat: str, long_term) -> Tuple[float, float]:
    now = time.time()
    if feat in _DYNAMIC and now - _DYNAMIC[feat][1] < _DYN_TTL:
        return _DYNAMIC[feat][0]
    rng = _calc_range(feat, long_term)
    if rng[0] == rng[1]:
        rng = DEFAULT_RANGES[feat]
    _DYNAMIC[feat] = (rng, now)
    return rng

# ───────────────────────────────────────────────
def summarise_past(trades, rng_p, rng_d):
    if not trades:
        return [0.5, 0.5]
    prof = [t.get("profit", 0) for t in trades]
    dur  = [t.get("duration", 0) for t in trades]
    return [normalize(np.mean(prof), rng_p),
            normalize(np.mean(dur),  rng_d)]

# ───────────────────────────────────────────────
# Simple regime classifier (rule‑based)
def _classify_regime(one_day: dict, vix_val: float) -> str:
    price   = one_day.get("price", 0)
    ema200  = one_day.get("ema_200", price)
    if price > ema200 and vix_val < 18:
        return "bull"
    if price < ema200 and vix_val > 25:
        return "bear"
    return "vol_cluster"

def _regime_one_hot(regime: str) -> List[float]:
    # bull → [1,0,0], bear → [0,1,0], vol_cluster → [0,0,1]
    if regime == "bull":
        return [1.0, 0.0, 0.0]
    if regime == "bear":
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]

# ───────────────────────────────────────────────
# ENTRY STATE
def build_meta_state_for_entry(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    confidence_score: float,
    trade_type: int,
    past_trades=None,
    long_term_data=None,
    position_size: float = 0.0,
) -> np.ndarray:

    past_trades   = past_trades or []
    long_term_data= long_term_data or {}

    try:
        # Dynamic ranges
        rsi_rng  = get_range("RSI",       long_term_data)
        macd_rng = get_range("MACD",      long_term_data)
        ema_rng  = get_range("EMA_DIST",  long_term_data)
        vol_rng  = get_range("VOL",       long_term_data)
        dur_rng  = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        # Per‑TF helper
        def tf_feats(df):
            last = df.iloc[-1]
            return [
                normalize(last.get("rsi", 50), rsi_rng),
                normalize(last.get("macd", 0), macd_rng),
                normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                normalize(last.get("volume", 0), vol_rng),
            ]

        # ── Regime classification
        vix_val = fetch_vix_price() or 20.0       # ← was get_vix_level()
        regime  = _classify_regime(data_1d.iloc[-1], vix_val)
        regime_vect = _regime_one_hot(regime)

        state = [
            normalize(confidence_score, DEFAULT_RANGES["CONF"]),
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            normalize(vix_val, DEFAULT_RANGES["VIX"]),

            # ‑‑ Position sizing
            normalize(position_size, DEFAULT_RANGES["SIZE"]),

            # ‑‑ Regime one‑hot
            *regime_vect,

            # ‑‑ Past trades summary
            *summarise_past(past_trades, prof_rng, dur_rng),

            # ‑‑ Time‑frame features
            *tf_feats(data_1m), *tf_feats(data_5m),
            *tf_feats(data_15m), *tf_feats(data_1h), *tf_feats(data_1d),
        ]

        # ‑‑ Long‑term trends
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

        return np.asarray(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"[MetaState] entry build error: {e}")
        # keep vector length stable
        return np.zeros(73, dtype=np.float32)   # 73 = previous 70 + 3 regime bits

# ───────────────────────────────────────────────
# Exit state (unchanged except VIX call)
_OPTION_CACHE: Dict[str, Tuple[dict, float]] = {}
def _cached_option_quote(sym: str, ttl=6):
    now = time.time()
    if sym in _OPTION_CACHE and now - _OPTION_CACHE[sym][1] < ttl:
        return _OPTION_CACHE[sym][0]
    q = get_option_quote(sym)
    _OPTION_CACHE[sym] = (q, now)
    return q

def build_meta_state_for_exit(
    trade: dict,
    past_trades=None,
    long_term_data=None,
) -> np.ndarray:
    past_trades   = past_trades or []
    long_term_data= long_term_data or {}
    try:
        spy_q      = get_spy_latest_quote() or {}
        spy_price  = spy_q.get("price", 0)

        opt_sym = trade.get("option_symbol")
        opt_q   = _cached_option_quote(opt_sym) if opt_sym else {}
        iv, delta = float(opt_q.get("iv", 0) or 0), float(opt_q.get("delta", 0) or 0)

        dur_rng   = DEFAULT_RANGES["DURATION"]
        prof_rng  = DEFAULT_RANGES["PROFIT"]
        spy_abs   = DEFAULT_RANGES["SPY_ABS"]

        confidence = trade.get("confidence", 0.5)
        t_type     = trade.get("trade_type", 0)
        entry      = trade.get("entry_price", 0)
        pnl_pct    = (spy_price - entry) / max(entry, 1e-9)

        minutes_open = normalize(
            (datetime.now(eastern) -
             datetime.fromisoformat(trade.get("timestamp")).astimezone(eastern)
             ).seconds // 60,
            dur_rng)

        state = [
            normalize(confidence, DEFAULT_RANGES["CONF"]),
            1.0 if t_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            minutes_open,
            normalize(pnl_pct, prof_rng),
            normalize(fetch_vix_price() or 20.0, DEFAULT_RANGES["VIX"]),  # ← updated
            *summarise_past(trades=past_trades, rng_p=prof_rng, rng_d=dur_rng),
            normalize(spy_price - entry, DEFAULT_RANGES["EMA_DIST"]),
            normalize(spy_price, spy_abs),
            normalize(iv, DEFAULT_RANGES["IV"]),
            normalize(delta, DEFAULT_RANGES["DELTA"]),
        ]
        return np.asarray(state, dtype=np.float32)
    except Exception as e:
        logger.error(f"[MetaState] exit build error: {e}")
        return np.zeros(43, dtype=np.float32)   # previous 40 + 3 regime bits (if added later)