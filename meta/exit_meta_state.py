import numpy as np
from datetime import datetime
from utils.vix_utils import get_vix_level
from utils.time_utils import minutes_since
from config import MARKET_OPEN

def build_exit_meta_state(trade: dict, option_data: dict, current_price: float) -> np.ndarray:
    """
    Construct state vector for exit decision using trade + current market/option info.

    trade: {
        'entry_time': '2025-06-07T10:12:00',
        'direction': 'call' or 'put',
        'entry_price': float,
        'entry_confidence': float,
        'duration': int (mins),
        ...
    }

    option_data: {
        'theta': float,
        'delta': float,
        'gamma': float,
        'vega': float,
        'bid': float,
        'ask': float,
        'last': float,
        ...
    }

    current_price: float – latest SPY price
    """
    now = datetime.utcnow()
    entry_time = datetime.fromisoformat(trade['entry_time'])
    minutes_held = max(minutes_since(entry_time), 1)

    # Greeks (can be None if missing)
    theta = option_data.get('theta', 0.0)
    delta = option_data.get('delta', 0.0)
    gamma = option_data.get('gamma', 0.0)
    vega = option_data.get('vega', 0.0)

    # Trade metrics
    entry_price = trade['entry_price']
    current_option_price = option_data.get('last') or (option_data.get('bid') + option_data.get('ask')) / 2
    pnl = (current_option_price - entry_price) / entry_price  # % return

    # VIX regime classification
    vix_level = get_vix_level()
    is_high_vix = 1.0 if vix_level == "high" else 0.0
    is_moderate_vix = 1.0 if vix_level == "moderate" else 0.0

    # Direction: 1 for call, 0 for put
    is_call = 1.0 if trade.get("direction") == "call" else 0.0

    # Entry confidence (if recorded)
    confidence = trade.get("entry_confidence", 0.5)

    # Normalized features
    norm_pnl = np.clip(pnl, -1, 3)          # cap large wins/losses
    norm_minutes = np.log1p(minutes_held)   # compress time range

    return np.array([
        norm_pnl,
        norm_minutes,
        theta,
        delta,
        gamma,
        vega,
        is_call,
        confidence,
        is_high_vix,
        is_moderate_vix,
    ], dtype=np.float32)