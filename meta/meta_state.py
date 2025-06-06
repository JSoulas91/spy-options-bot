# meta/meta_state.py
import numpy as np
from datetime import datetime
import pytz

from utils.vix_utils import get_vix_level
from utils.logger import bot_logger as logger

eastern = pytz.timezone("US/Eastern")

# ───────────────────────────────
# Normalization helper (with clamp)
# ───────────────────────────────
def normalize(value, min_val, max_val):
    norm = (value - min_val) / (max_val - min_val + 1e-9)
    # hard‑clip to [0,1] so extreme values don’t distort the state
    return max(0.0, min(1.0, norm))

# Fixed ranges (one spot to tune later)
RANGES = {
    "RSI":   (0,   100),
    "MACD":  (-10,  10),
    "PRICE_EMA_DIFF": (-20, 20),   # price - EMA20
    "VOLUME": (1e3, 2e7),          # raw share volume
    "VIX":   (10,  40),
    "PROFIT":(-1.0, 1.0),
    "DURATION": (0, 390),          # minutes
}

# ───────────────────────────────
# Helper: minutes since US market open
# ───────────────────────────────
def get_minutes_since_open():
    now = datetime.now(eastern)
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, (now - open_time).seconds // 60)

# ───────────────────────────────
# Confluence score across timeframes
# ───────────────────────────────
def compute_multi_timeframe_confluence(*data_frames):
    score, total = 0, 0
    for data in data_frames:
        if data.get("rsi", 50) > 50: score += 1
        if data.get("macd", 0)  >  0: score += 1
        if data.get("price", 0) > data.get("ema_20", 0): score += 1
        total += 3
    return score / total if total else 0.0

# ───────────────────────────────
# Feature extraction helpers
# ───────────────────────────────
def extract_features(data):
    return [
        normalize(data.get("rsi", 50), *RANGES["RSI"]),
        normalize(data.get("macd", 0), *RANGES["MACD"]),
        normalize(data.get("price", 0) - data.get("ema_20", 0),
                  *RANGES["PRICE_EMA_DIFF"]),
        normalize(data.get("volume", 1e5), *RANGES["VOLUME"]),
    ]

def extract_binary_trend(data):
    return [
        normalize(data.get("rsi", 50), *RANGES["RSI"]),
        normalize(data.get("macd", 0), *RANGES["MACD"]),
        1.0 if data.get("price", 0) > data.get("ema_20", 0) else 0.0,
    ]

def extract_long_term_trend(long_term_data):
    periods = ["5d", "10d", "15d", "1mo", "3mo", "6mo"]
    feats = []
    for p in periods:
        d = long_term_data.get(p, {})
        feats += [
            normalize(d.get("rsi", 50), *RANGES["RSI"]),
            normalize(d.get("macd", 0), *RANGES["MACD"]),
            normalize(d.get("price", 0) - d.get("ema_20", 0),
                      *RANGES["PRICE_EMA_DIFF"]),
        ]
    return feats

def summarize_past_trades(past_trades):
    if not past_trades:
        return [0.5, 0.5]
    avg_profit   = np.mean([t.get("profit", 0)   for t in past_trades])
    avg_duration = np.mean([t.get("duration", 0) for t in past_trades])
    return [
        normalize(avg_profit,   *RANGES["PROFIT"]),
        normalize(avg_duration, *RANGES["DURATION"]),
    ]

# ───────────────────────────────
# Entry meta‑state
# ───────────────────────────────
def build_meta_state_for_entry(data_1m, data_5m, data_15m, data_1h, data_1d,
                               confidence_score, trade_type,
                               past_trades=None, long_term_data=None):
    past_trades   = past_trades   or []
    long_term_data = long_term_data or {}
    try:
        state = [
            confidence_score,
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), *RANGES["DURATION"]),
            normalize(get_vix_level(), *RANGES["VIX"]),
            *summarize_past_trades(past_trades),
            *extract_features(data_1m),
            *extract_binary_trend(data_5m),
            *extract_binary_trend(data_15m),
            *extract_binary_trend(data_1h),
            *extract_binary_trend(data_1d),
            *extract_long_term_trend(long_term_data),
            compute_multi_timeframe_confluence(
                data_1m, data_5m, data_15m, data_1h, data_1d
            ),
        ]
        return np.array(state, dtype=np.float32)
    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_entry: {e}")
        # Keep fixed length so model input size doesn't change
        return np.zeros( len(RANGES)*5 , dtype=np.float32)

# ───────────────────────────────
# Exit meta‑state
# ───────────────────────────────
def build_meta_state_for_exit(data_1m, data_5m, data_15m, data_1h, data_1d,
                              confidence_score, trade_type,
                              trade_duration_minutes, current_profit,
                              past_trades=None, long_term_data=None):
    past_trades   = past_trades   or []
    long_term_data = long_term_data or {}
    try:
        state = [
            confidence_score,
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), *RANGES["DURATION"]),
            normalize(trade_duration_minutes, *RANGES["DURATION"]),
            normalize(current_profit, *RANGES["PROFIT"]),
            normalize(get_vix_level(), *RANGES["VIX"]),
            *summarize_past_trades(past_trades),
            *extract_features(data_1m),
            *extract_binary_trend(data_5m),
            *extract_binary_trend(data_15m),
            *extract_binary_trend(data_1h),
            *extract_binary_trend(data_1d),
            *extract_long_term_trend(long_term_data),
            compute_multi_timeframe_confluence(
                data_1m, data_5m, data_15m, data_1h, data_1d
            ),
        ]
        return np.array(state, dtype=np.float32)
    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_exit: {e}")
        return np.zeros( len(RANGES)*5 + 5 , dtype=np.float32)