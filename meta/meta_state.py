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
PAD_VAL = 0.50

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
    dur = [t.get("duration", 0) for t in trades]
    return [normalize(np.mean(prof), rng_p),
            normalize(np.mean(dur), rng_d)]


def _pad(vec: List[float]) -> np.ndarray:
    if len(vec) > STATE_DIM:
        vec = vec[:STATE_DIM]
    else:
        vec += [PAD_VAL] * (STATE_DIM - len(vec))
    return np.asarray(vec, dtype=np.float32)


def _classify_regime(one_day: dict, vix_val: float) -> str:
    price = one_day.get("price", 0)
    ema200 = one_day.get("ema_200", price)
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
    position_size: float = 0.0,
    trade_type: int = 1,
    confidence_score: float = 0.5,
    past_trades=None,
    long_term_data=None,
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    import pandas as pd
    import numpy as np
    past_trades = past_trades or []
    long_term_data = long_term_data or {}

    def ensure_df(df):
        if isinstance(df, dict):
            # Check if any value is list-like -> treat as tabular data
            if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in df.values()):
                return pd.DataFrame(df)
            else:
                # All scalars: create single-row DataFrame
                return pd.DataFrame([df])
        return df

    data_1m = ensure_df(data_1m)
    data_5m = ensure_df(data_5m)
    data_15m = ensure_df(data_15m)
    data_1h = ensure_df(data_1h)
    data_1d = ensure_df(data_1d)

    try:
        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            if df is None:
                return [
                    normalize(50, rsi_rng),
                    normalize(0, macd_rng),
                    normalize(0, ema_rng),
                    normalize(0, vol_rng),
                ]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            if isinstance(df, dict):
                rsi = df.get("rsi", 50)
                macd = df.get("macd", 0)
                price = df.get("price", 0)
                ema_20 = df.get("ema_20", 0)
                volume = df.get("volume", 0)
                return [
                    normalize(rsi, rsi_rng),
                    normalize(macd, macd_rng),
                    normalize(price - ema_20, ema_rng),
                    normalize(volume, vol_rng),
                ]
            return [
                normalize(50, rsi_rng),
                normalize(0, macd_rng),
                normalize(0, ema_rng),
                normalize(0, vol_rng),
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
            # 1. Success probability (trade success prob)
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            # 2. Predicted direction: one-hot for 3 classes (long, neutral, short)
            pred_dir = classifier_output.get("predicted_direction", -1)
            if pred_dir in (0, 1, 2):
                dir_one_hot = [0, 0, 0]
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL, PAD_VAL, PAD_VAL]
            state.extend(dir_one_hot)
            # 3. Class probabilities (3 floats)
            class_probs = classifier_output.get("class_probabilities", [PAD_VAL, PAD_VAL, PAD_VAL])
            if len(class_probs) != 3:
                class_probs = [PAD_VAL, PAD_VAL, PAD_VAL]
            state.extend(class_probs)
            # 4. Entropy (normalized 0 to 1)
            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            state += [PAD_VAL] * 8

        return _pad(state)

    except Exception as e:
        logger.error(f"Error building meta state for entry: {e}")
        return _pad([PAD_VAL] * STATE_DIM)


def build_meta_state_for_exit(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    position_size: float = 0.0,
    trade_type: int = 1,
    confidence_score: float = 0.5,
    past_trades=None,
    long_term_data=None,
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    import pandas as pd
    import numpy as np
    past_trades = past_trades or []
    long_term_data = long_term_data or {}

    def ensure_df(df):
        if isinstance(df, dict):
            if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in df.values()):
                return pd.DataFrame(df)
            else:
                return pd.DataFrame([df])
        return df

    data_1m = ensure_df(data_1m)
    data_5m = ensure_df(data_5m)
    data_15m = ensure_df(data_15m)
    data_1h = ensure_df(data_1h)
    data_1d = ensure_df(data_1d)

    try:
        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            if df is None:
                return [
                    normalize(50, rsi_rng),
                    normalize(0, macd_rng),
                    normalize(0, ema_rng),
                    normalize(0, vol_rng),
                ]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            if isinstance(df, dict):
                rsi = df.get("rsi", 50)
                macd = df.get("macd", 0)
                price = df.get("price", 0)
                ema_20 = df.get("ema_20", 0)
                volume = df.get("volume", 0)
                return [
                    normalize(rsi, rsi_rng),
                    normalize(macd, macd_rng),
                    normalize(price - ema_20, ema_rng),
                    normalize(volume, vol_rng),
                ]
            return [
                normalize(50, rsi_rng),
                normalize(0, macd_rng),
                normalize(0, ema_rng),
                normalize(0, vol_rng),
            ]

        vix_val = fetch_vix_price() or 20.0
        regime = classifier_output.get("regime_class") if classifier_output and "regime_class" in classifier_output else _classify_regime(data_1d.iloc[-1], vix_val)

        clf_conf = classifier_output.get("trade_success_prob") if classifier_output else None
        norm_conf = normalize(clf_conf if clf
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

        if classifier_output:
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            pred_dir = classifier_output.get("predicted_direction", -1)
            if pred_dir in (0, 1, 2):
                dir_one_hot = [0, 0, 0]
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL, PAD_VAL, PAD_VAL]
            state.extend(dir_one_hot)
            class_probs = classifier_output.get("class_probabilities", [PAD_VAL, PAD_VAL, PAD_VAL])
        if len(class_probs) != 3:
            class_probs = [PAD_VAL, PAD_VAL, PAD_VAL]
            state.extend(class_probs)

            # 4. Entropy (normalized 0 to 1)
            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            # No classifier output, pad with PAD_VAL for 8 features
            state += [PAD_VAL] * 8

        return _pad(state)

    except Exception as e:
        logger.error(f"Error building meta state for exit: {e}")
        return _pad([PAD_VAL] * STATE_DIM)