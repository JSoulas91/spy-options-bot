import numpy as np
from utils.vix_utils import get_vix_level  # Ensure this utility exists or mock it

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-9)

def compute_indicator_confluence(market_data):
    score = 0
    total = 3

    rsi = market_data.get("rsi", 50)
    macd = market_data.get("macd", 0)
    price = market_data.get("price", 0)
    ema_20 = market_data.get("ema_20", 0)

    if rsi > 50:
        score += 1
    if macd > 0:
        score += 1
    if price > ema_20:
        score += 1

    return score / total

def build_meta_state_for_entry(market_data, confidence_score, trade_type):
    state = []

    state.append(confidence_score)
    state.append(1.0 if trade_type == 1 else 0.0)

    vix = get_vix_level()
    state.append(normalize(vix, 10, 40))

    price = market_data.get("price", 0)
    ema_20 = market_data.get("ema_20", 0)
    rsi = market_data.get("rsi", 50)
    macd = market_data.get("macd", 0)
    volume = market_data.get("volume", 1)

    state.append(normalize(rsi, 0, 100))
    state.append(normalize(macd, -5, 5))
    state.append(normalize(price - ema_20, -10, 10))
    state.append(normalize(volume, 1e4, 1e7))

    # ✅ Add confluence score
    confluence_score = compute_indicator_confluence(market_data)
    state.append(confluence_score)

    return np.array(state, dtype=np.float32)

def build_meta_state_for_exit(market_data, confidence_score, trade_type, trade_duration_minutes, current_profit):
    state = []

    state.append(confidence_score)
    state.append(1.0 if trade_type == 1 else 0.0)
    state.append(normalize(trade_duration_minutes, 0, 390))
    state.append(normalize(current_profit, -1.0, 1.0))

    vix = get_vix_level()
    state.append(normalize(vix, 10, 40))

    rsi = market_data.get("rsi", 50)
    macd = market_data.get("macd", 0)
    ema_diff = market_data.get("price", 0) - market_data.get("ema_20", 0)
    volume = market_data.get("volume", 1)

    state.append(normalize(rsi, 0, 100))
    state.append(normalize(macd, -5, 5))
    state.append(normalize(ema_diff, -10, 10))
    state.append(normalize(volume, 1e4, 1e7))

    # ✅ Add confluence score
    confluence_score = compute_indicator_confluence(market_data)
    state.append(confluence_score)

    return np.array(state, dtype=np.float32)