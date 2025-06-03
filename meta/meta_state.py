# meta/meta_state.py

import numpy as np
from datetime import datetime
from utils.vix_utils import get_vix_level
from utils.logger import bot_logger as logger
import pytz

eastern = pytz.timezone("US/Eastern")

# --- Normalization helper ---
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-9)

# --- Time-of-day helper ---
def get_minutes_since_open():
    now = datetime.now(eastern)
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return max(0, (now - open_time).seconds // 60)

# --- Confluence score computation ---
def compute_multi_timeframe_confluence(*data_frames):
    score = 0
    total = 0
    for data in data_frames:
        if data.get("rsi", 50) > 50: score += 1
        if data.get("macd", 0) > 0: score += 1
        if data.get("price", 0) > data.get("ema_20", 0): score += 1
        total += 3
    return score / total if total > 0 else 0.0

# --- Feature extraction ---
def extract_features(data):
    return [
        normalize(data.get("rsi", 50), 0, 100),
        normalize(data.get("macd", 0), -10, 10),
        normalize(data.get("price", 0) - data.get("ema_20", 0), -20, 20),
        normalize(data.get("volume", 1e5), 1e3, 2e7),
    ]

def extract_binary_trend(data):
    return [
        normalize(data.get("rsi", 50), 0, 100),
        normalize(data.get("macd", 0), -10, 10),
        1.0 if data.get("price", 0) > data.get("ema_20", 0) else 0.0
    ]

# --- Long-term trend feature extractor ---
def extract_long_term_trend(long_term_data):
    """
    Expects a dictionary with keys: '5d', '10d', '15d', '1mo', '3mo', '6mo'
    Each key maps to a data dict with 'rsi', 'macd', 'price', 'ema_20'
    """
    features = []
    periods = ['5d', '10d', '15d', '1mo', '3mo', '6mo']
    for period in periods:
        data = long_term_data.get(period, {})
        features += [
            normalize(data.get("rsi", 50), 0, 100),
            normalize(data.get("macd", 0), -10, 10),
            normalize(data.get("price", 0) - data.get("ema_20", 0), -20, 20),
        ]
    return features

# --- Temporal memory summary ---
def summarize_past_trades(past_trades: list):
    if not past_trades:
        return [0.5, 0.5]  # neutral defaults

    avg_profit = np.mean([t.get("profit", 0) for t in past_trades])
    avg_duration = np.mean([t.get("duration", 0) for t in past_trades])

    return [
        normalize(avg_profit, -1.0, 1.0),
        normalize(avg_duration, 0, 390),
    ]

# --- Entry Meta State Builder ---
def build_meta_state_for_entry(data_1m, data_5m, data_15m, data_1h, data_1d,
                                confidence_score, trade_type, past_trades=[],
                                long_term_data={}):
    try:
        state = []

        # Contextual
        state.append(confidence_score)
        state.append(1.0 if trade_type == 1 else 0.0)
        state.append(normalize(get_minutes_since_open(), 0, 390))

        # VIX
        vix = get_vix_level()
        state.append(normalize(vix, 10, 40))

        # Temporal memory
        state += summarize_past_trades(past_trades)

        # Timeframe Features
        state += extract_features(data_1m)
        state += extract_binary_trend(data_5m)
        state += extract_binary_trend(data_15m)
        state += extract_binary_trend(data_1h)
        state += extract_binary_trend(data_1d)

        # Long-Term Features
        state += extract_long_term_trend(long_term_data)

        # Confluence
        confluence = compute_multi_timeframe_confluence(data_1m, data_5m, data_15m, data_1h, data_1d)
        state.append(confluence)

        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_entry: {e}")
        return np.zeros(42, dtype=np.float32)

# --- Exit Meta State Builder ---
def build_meta_state_for_exit(data_1m, data_5m, data_15m, data_1h, data_1d,
                               confidence_score, trade_type, trade_duration_minutes,
                               current_profit, past_trades=[], long_term_data={}):
    try:
        state = []

        # Contextual
        state.append(confidence_score)
        state.append(1.0 if trade_type == 1 else 0.0)
        state.append(normalize(get_minutes_since_open(), 0, 390))
        state.append(normalize(trade_duration_minutes, 0, 390))
        state.append(normalize(current_profit, -1.0, 1.0))

        # VIX
        vix = get_vix_level()
        state.append(normalize(vix, 10, 40))

        # Temporal memory
        state += summarize_past_trades(past_trades)

        # Timeframe Features
        state += extract_features(data_1m)
        state += extract_binary_trend(data_5m)
        state += extract_binary_trend(data_15m)
        state += extract_binary_trend(data_1h)
        state += extract_binary_trend(data_1d)

        # Long-Term Features
        state += extract_long_term_trend(long_term_data)

        # Confluence
        confluence = compute_multi_timeframe_confluence(data_1m, data_5m, data_15m, data_1h, data_1d)
        state.append(confluence)

        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_exit: {e}")
        return np.zeros(45, dtype=np.float32)