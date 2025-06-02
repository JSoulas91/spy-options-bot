import numpy as np
from utils.vix_utils import get_vix_level
from utils.logger import bot_logger as logger

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-9)

def compute_multi_timeframe_confluence(*data_frames):
    """Score confluence across multiple timeframe data dictionaries."""
    score = 0
    total = 0

    for data in data_frames:
        if data.get("rsi", 50) > 50:
            score += 1
        if data.get("macd", 0) > 0:
            score += 1
        if data.get("price", 0) > data.get("ema_20", 0):
            score += 1
        total += 3

    return score / total if total > 0 else 0.0

def extract_features(data):
    return [
        normalize(data.get("rsi", 50), 0, 100),
        normalize(data.get("macd", 0), -5, 5),
        normalize(data.get("price", 0) - data.get("ema_20", 0), -10, 10),
        normalize(data.get("volume", 1e5), 1e4, 1e7)
    ]

def extract_binary_trend(data):
    return [
        normalize(data.get("rsi", 50), 0, 100),
        normalize(data.get("macd", 0), -5, 5),
        1.0 if data.get("price", 0) > data.get("ema_20", 0) else 0.0
    ]

def build_meta_state_for_entry(data_1m, data_5m, data_15m, data_1h, data_1d, confidence_score, trade_type):
    try:
        state = []

        # Contextual
        state.append(confidence_score)
        state.append(1.0 if trade_type == 1 else 0.0)

        # VIX
        vix = get_vix_level()
        state.append(normalize(vix, 10, 40))

        # 1m
        state += extract_features(data_1m)

        # 5m
        state += extract_binary_trend(data_5m)

        # 15m
        state += extract_binary_trend(data_15m)

        # 1h
        state += extract_binary_trend(data_1h)

        # 1d
        state += extract_binary_trend(data_1d)

        # Confluence Score
        confluence = compute_multi_timeframe_confluence(data_1m, data_5m, data_15m, data_1h, data_1d)
        state.append(confluence)

        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_entry: {e}")
        return np.zeros(22, dtype=np.float32)

def build_meta_state_for_exit(data_1m, data_5m, data_15m, data_1h, data_1d, confidence_score, trade_type, trade_duration_minutes, current_profit):
    try:
        state = []

        # Contextual
        state.append(confidence_score)
        state.append(1.0 if trade_type == 1 else 0.0)
        state.append(normalize(trade_duration_minutes, 0, 390))
        state.append(normalize(current_profit, -1.0, 1.0))

        # VIX
        vix = get_vix_level()
        state.append(normalize(vix, 10, 40))

        # 1m
        state += extract_features(data_1m)

        # 5m
        state += extract_binary_trend(data_5m)

        # 15m
        state += extract_binary_trend(data_15m)

        # 1h
        state += extract_binary_trend(data_1h)

        # 1d
        state += extract_binary_trend(data_1d)

        # Confluence Score
        confluence = compute_multi_timeframe_confluence(data_1m, data_5m, data_15m, data_1h, data_1d)
        state.append(confluence)

        return np.array(state, dtype=np.float32)

    except Exception as e:
        logger.error(f"❌ Error in build_meta_state_for_exit: {e}")
        return np.zeros(24, dtype=np.float32)