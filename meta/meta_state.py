import numpy as np
from utils.vix_utils import get_vix_level  # Ensure this utility exists or mock it

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-9)

def build_meta_state_for_entry(market_data, confidence_score, trade_type):
    """
    Builds the meta-agent state vector for trade entry decisions.
    """
    state = []

    # Normalize confidence score (expected range: 0 to 1)
    state.append(confidence_score)

    # Trade type: 0 = day trade, 1 = swing trade
    state.append(1.0 if trade_type == 1 else 0.0)

    # Volatility level (e.g., normalized VIX)
    vix = get_vix_level()
    state.append(normalize(vix, 10, 40))  # Example range

    # Example technical indicators (you can pull more features from market_data if needed)
    price = market_data.get("price", 0)
    ema_20 = market_data.get("ema_20", 0)
    rsi = market_data.get("rsi", 50)
    macd = market_data.get("macd", 0)
    volume = market_data.get("volume", 1)

    state.append(normalize(rsi, 0, 100))
    state.append(normalize(macd, -5, 5))
    state.append(normalize(price - ema_20, -10, 10))
    state.append(normalize(volume, 1e4, 1e7))

    return np.array(state, dtype=np.float32)

def build_meta_state_for_exit(market_data, confidence_score, trade_type, trade_duration_minutes, current_profit):
    """
    Builds the meta-agent state vector for trade exit decisions.
    """
    state = []

    # Confidence level at time of evaluation
    state.append(confidence_score)

    # Trade type: 0 = day, 1 = swing
    state.append(1.0 if trade_type == 1 else 0.0)

    # Trade duration normalized (e.g., up to 390 mins in a day)
    state.append(normalize(trade_duration_minutes, 0, 390))

    # Current profit (expected -1.0 to +1.0, but normalize just in case)
    state.append(normalize(current_profit, -1.0, 1.0))

    # VIX/volatility
    vix = get_vix_level()
    state.append(normalize(vix, 10, 40))

    # Additional indicators at exit time
    rsi = market_data.get("rsi", 50)
    macd = market_data.get("macd", 0)
    ema_diff = market_data.get("price", 0) - market_data.get("ema_20", 0)
    volume = market_data.get("volume", 1)

    state.append(normalize(rsi, 0, 100))
    state.append(normalize(macd, -5, 5))
    state.append(normalize(ema_diff, -10, 10))
    state.append(normalize(volume, 1e4, 1e7))

    return np.array(state, dtype=np.float32)