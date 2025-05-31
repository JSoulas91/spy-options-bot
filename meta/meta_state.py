# meta/meta_state.py

import os
from datetime import datetime
from utils.vix_utils import get_current_vix
from utils.trade_history import get_recent_trade_results
from config import META_STATE_LOOKBACK_MINUTES

def get_time_of_day_bucket():
    now = datetime.now().time()
    if now < datetime.strptime("11:00", "%H:%M").time():
        return 0  # Morning
    elif now < datetime.strptime("14:00", "%H:%M").time():
        return 1  # Midday
    else:
        return 2  # Afternoon

def compute_meta_state(trade_type: str):
    """
    Returns the current meta-state vector:
    [win/loss streak, VIX, time of day bucket (0/1/2), trade_type (0=day, 1=swing), volatility estimate]
    """
    # 1. Win/Loss streak
    recent_results = get_recent_trade_results(lookback_minutes=META_STATE_LOOKBACK_MINUTES)
    streak = 0
    for result in reversed(recent_results):
        if result == 'win':
            streak = streak + 1 if streak >= 0 else 1
        elif result == 'loss':
            streak = streak - 1 if streak <= 0 else -1
        else:
            break

    # 2. VIX
    vix = get_current_vix() or 18.0

    # 3. Time of Day
    time_bucket = get_time_of_day_bucket()

    # 4. Trade Type
    trade_type_val = 0 if trade_type == "day" else 1

    # 5. Volatility estimate (using VIX as proxy for now)
    volatility = vix / 100.0

    return [streak, vix, time_bucket, trade_type_val, volatility]