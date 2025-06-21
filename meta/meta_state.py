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

def summarise_past(trades: List[Dict], rng_p, rng_d) -> List[float]:
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

def pad_sequence(seq: List[np.ndarray]) -> np.ndarray:
    """Pad/truncate a sequence to (STATE_SEQUENCE_LENGTH, STATE_DIM)."""
    seq_len = len(seq)
    if seq_len >= STATE_SEQUENCE_LENGTH:
        return np.stack(seq[-STATE_SEQUENCE_LENGTH:])
    pad_count = STATE_SEQUENCE_LENGTH - seq_len
    padding = [np.full(STATE_DIM, PAD_VAL, dtype=np.float32)] * pad_count
    return np.stack(padding + seq)

def _normalize_features(features: List[Dict], long_term: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Normalize each timestep dict into a state vector."""
    out = []
    for f in features:
        vec = []

        # Normalize core indicators
        for key in ["RSI", "MACD", "VOL", "VIX", "SPY_ABS", "IV", "DELTA", "SIZE"]:
            val = f.get(key.lower(), 0)
            rng = get_range(key, long_term)
            vec.append(normalize(val, rng))

        # Normalize EMA distance
        if "price" in f and "ema_20" in f:
            ema_dist = f["price"] - f["ema_20"]
        else:
            ema_dist = 0
        vec.append(normalize(ema_dist, get_range("EMA_DIST", long_term)))

        # Add regime one-hot
        regime = _classify_regime(f, f.get("vix", 20))
        vec.extend(_regime_one_hot(regime))

        # Add classifier outputs (if present)
        vec.append(f.get("clf_prob", 0.5))           # success probability
        vec.extend(f.get("clf_dir", [0.5, 0.5]))      # one-hot direction
        vec.extend(f.get("clf_probs", [0.33, 0.33, 0.33]))  # full class probs
        vec.append(f.get("clf_entropy", 1.0))         # entropy

        # Add past trade summary
        trades = f.get("past_trades", [])
        prof_rng = get_range("PROFIT", long_term)
        dur_rng = get_range("DURATION", long_term)
        vec.extend(summarise_past(trades, prof_rng, dur_rng))

        # Pad to STATE_DIM
        out.append(_pad(vec))

    return out

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
            if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in df.values()):
                return pd.DataFrame(df)
            else:
                return pd.DataFrame([df])
        return df

    def build_sequence(state: List[float]) -> np.ndarray:
        padded = _pad(state)
        return np.stack([padded.copy() for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)

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
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]

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
                state += [PAD_VAL] * 3

        if classifier_output:
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            pred_dir = classifier_output.get("predicted_direction", -1)
            dir_one_hot = [0.0, 0.0, 0.0]
            if pred_dir in (0, 1, 2):
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL] * 3
            state.extend(dir_one_hot)

            class_probs = classifier_output.get("class_probabilities", [PAD_VAL] * 3)
            if len(class_probs) != 3:
                class_probs = [PAD_VAL] * 3
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            state += [PAD_VAL] * 8

        return build_sequence(state)

    except Exception as e:
        logger.error(f"Error building meta state for entry: {e}")
        return np.stack([_pad([PAD_VAL] * STATE_DIM) for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)


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

    def build_sequence(state: List[float]) -> np.ndarray:
        padded = _pad(state)
        return np.stack([padded.copy() for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)

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
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]

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
                state += [PAD_VAL] * 3

        if classifier_output:
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            pred_dir = classifier_output.get("predicted_direction", -1)
            dir_one_hot = [0.0, 0.0, 0.0]
            if pred_dir in (0, 1, 2):
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL] * 3
            state.extend(dir_one_hot)

            class_probs = classifier_output.get("class_probabilities", [PAD_VAL] * 3)
            if len(class_probs) != 3:
                class_probs = [PAD_VAL] * 3
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            state += [PAD_VAL] * 8

        return build_sequence(state)

    except Exception as e:
        logger.error(f"Error building meta state for exit: {e}")
        return np.stack([_pad([PAD_VAL] * STATE_DIM) for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)