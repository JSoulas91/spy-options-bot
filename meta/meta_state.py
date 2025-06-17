from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pytz

from utils.logger import bot_logger as logger
from utils.vix_utils import fetch_vix_price
from data.multi_timeframe_fetcher import (
    get_minutes_since_open,
    get_spy_latest_quote,
)
from data.options_fetcher import get_quote as get_option_quote
from config import MAX_POSITION_SIZE

# ───────────────────────────────────────────────
eastern = pytz.timezone("US/Eastern")

# Expanded STATE_DIM from 78 to 83 to add classifier outputs for both entry and exit
STATE_DIM = 83
PAD_VAL   = 0.50

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
    if hi - lo == 0:
        return 0.5
    return float(max(0, min(1, (val - lo) / (hi - lo))))

_DYNAMIC: Dict[str, Tuple[Tuple[float, float], float]] = {}
_DYN_TTL = 3600

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

def summarise_past(trades, rng_p, rng_d):
    if not trades:
        return [0.5, 0.5]
    prof = [t.get("profit", 0) for t in trades]
    dur  = [t.get("duration", 0) for t in trades]
    return [normalize(np.mean(prof), rng_p),
            normalize(np.mean(dur),  rng_d)]

def _pad(vec: List[float]) -> np.ndarray:
    if len(vec) > STATE_DIM:
        vec = vec[:STATE_DIM]
    else:
        vec += [PAD_VAL] * (STATE_DIM - len(vec))
    return np.asarray(vec, dtype=np.float32)

def _classify_regime(one_day: dict, vix_val: float) -> str:
    price   = one_day.get("price", 0)
    ema200  = one_day.get("ema_200", price)
    if price > ema200 and vix_val < 18:
        return "bull"
    if price < ema200 and vix_val > 25:
        return "bear"
    return "vol_cluster"

def _regime_one_hot(regime: str) -> List[float]:
    if regime == "bull":
        return [1.0, 0.0, 0.0]
    if regime == "bear":
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]

def build_meta_state_for_entry(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    confidence_score: float,
    trade_type: int,
    past_trades=None,
    long_term_data=None,
    position_size: float = 0.0,
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    past_trades   = past_trades or []
    long_term_data= long_term_data or {}

    try:
        rsi_rng  = get_range("RSI",       long_term_data)
        macd_rng = get_range("MACD",      long_term_data)
        ema_rng  = get_range("EMA_DIST",  long_term_data)
        vol_rng  = get_range("VOL",       long_term_data)
        dur_rng  = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            last = df.iloc[-1]
            return [
                normalize(last.get("rsi", 50), rsi_rng),
                normalize(last.get("macd", 0), macd_rng),
                normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                normalize(last.get("volume", 0), vol_rng),
            ]

        vix_val = fetch_vix_price() or 20.0

        regime = classifier_output.get("regime_class") if classifier_output and "regime_class" in classifier_output else _classify_regime(data_1d.iloc[-1], vix_val)

        clf_conf = classifier_output.get("trade_success_prob") if classifier_output else None
        norm_conf = normalize(clf_conf if clf_conf is not None else confidence_score, DEFAULT_RANGES["CONF"])

        state: List[float] = [
            norm_conf,
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            normalize(vix_val, DEFAULT_RANGES["VIX"]),
            normalize(position_size, DEFAULT_RANGES["SIZE"]),
            *_regime_one_hot(regime),
            *summarise_past(past_trades, prof_rng, dur_rng),
            *tf_feats(data_1m), *tf_feats(data_5m),
            *tf_feats(data_15m), *tf_feats(data_1h), *tf_feats(data_1d),
        ]

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
                state += [PAD_VAL, PAD_VAL, PAD_VAL]

        # ───── Additional classifier outputs ─────
        if classifier_output:
            # 1. Success prob (already added above but appended here for agent emphasis)
            state.append(norm_conf)

            # 2. Predicted direction: -1 = put, 0 = neutral, 1 = call
            direction = classifier_output.get("predicted_direction", 0)
            # normalize to [0,1]
            state.append((direction + 1) / 2)

            # 3. Class probs (put, neutral, call)
            probs = classifier_output.get("class_probs", [1/3, 1/3, 1/3])
            if len(probs) == 3:
                state += probs
            else:
                state += [1/3, 1/3, 1/3]

            # 4. Confidence entropy
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)
            state.append(normalize(entropy, (0, 1.5)))

        return _pad(state)

    except Exception as e:
        logger.error(f"[MetaState] entry build error: {e}")
        return np.full(STATE_DIM, PAD_VAL, dtype=np.float32)

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
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    past_trades   = past_trades or []
    long_term_data= long_term_data or {}

    try:
        prof_rng = DEFAULT_RANGES["PROFIT"]
        dur_rng  = DEFAULT_RANGES["DURATION"]
        rsi_rng  = get_range("RSI",       long_term_data)
        macd_rng = get_range("MACD",      long_term_data)
        ema_rng  = get_range("EMA_DIST",  long_term_data)
        vol_rng  = get_range("VOL",       long_term_data)
        vix_val  = fetch_vix_price() or 20.0

        regime = classifier_output.get("regime_class") if classifier_output and "regime_class" in classifier_output else _classify_regime({}, vix_val)

        position_size = trade.get("size", 0)
        profit = trade.get("profit", 0)
        duration = trade.get("duration", 0)
        exit_type = trade.get("exit_type", 0)

        # Basic exit state features
        state = [
            normalize(profit, prof_rng),
            normalize(duration, dur_rng),
            normalize(position_size, DEFAULT_RANGES["SIZE"]),
            normalize(vix_val, DEFAULT_RANGES["VIX"]),
            *_regime_one_hot(regime),
        ]

        # Option features if available
        option_symbol = trade.get("option_symbol")
        if option_symbol:
            q = _cached_option_quote(option_symbol)
            if q:
                iv = q.get("implied_volatility", 0.0)
                delta = q.get("delta", 0.0)
                state += [
                    normalize(iv, DEFAULT_RANGES["IV"]),
                    normalize(delta, DEFAULT_RANGES["DELTA"]),
                ]
            else:
                state += [PAD_VAL, PAD_VAL]
        else:
            state += [PAD_VAL, PAD_VAL]

        # Summarise past trades features
        state += summarise_past(past_trades, prof_rng, dur_rng)

        # Exit type (one-hot or scalar)
        state.append(float(exit_type))

        # ───── Additional classifier outputs ─────
        if classifier_output:
            # 1. Success prob
            clf_conf = classifier_output.get("trade_success_prob")
            norm_conf = normalize(clf_conf if clf_conf is not None else 0.5, DEFAULT_RANGES["CONF"])
            state.append(norm_conf)

            # 2. Predicted direction: -1 = put, 0 = neutral, 1 = call → normalized [0,1]
            direction = classifier_output.get("predicted_direction", 0)
            state.append((direction + 1) / 2)

            # 3. Class probs (put, neutral, call)
            probs = classifier_output.get("class_probs", [1/3, 1/3, 1/3])
            if len(probs) == 3:
                state += probs
            else:
                state += [1/3, 1/3, 1/3]

            # 4. Confidence entropy
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)
            state.append(normalize(entropy, (0, 1.5)))

        return _pad(state)

    except Exception as e:
        logger.error(f"[MetaState] exit build error: {e}")
        return np.full(STATE_DIM, PAD_VAL, dtype=np.float32)