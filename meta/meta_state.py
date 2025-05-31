# meta/meta_state.py

from datetime import datetime
from utils.vix_utils import get_current_vix
from utils.trade_history import get_recent_trade_results
from config import META_STATE_LOOKBACK_MINUTES

def get_time_of_day_bucket():
    """
    Categorize time into one of 3 buckets:
    0 = Morning (before 11 AM)
    1 = Midday (11 AM to 2 PM)
    2 = Afternoon (after 2 PM)
    """
    now = datetime.now().time()
    if now < datetime.strptime("11:00", "%H:%M").time():
        return 0
    elif now < datetime.strptime("14:00", "%H:%M").time():
        return 1
    else:
        return 2

def compute_meta_state(trade_type: str):
    """
    Compute current meta-agent state vector.

    Args:
        trade_type (str): 'day' or 'swing'

    Returns:
        list: [streak, vix, time_bucket, trade_type_val, volatility]
    """
    # 1. Compute Win/Loss Streak from Recent History
    recent_results = get_recent_trade_results(lookback_minutes=META_STATE_LOOKBACK_MINUTES)
    streak = 0
    for result in reversed(recent_results):
        if result == 'win':
            streak = streak + 1 if streak >= 0 else 1
        elif result == 'loss':
            streak = streak - 1 if streak <= 0 else -1
        else:
            break  # stop at 'neutral' or unknown result

    # 2. Get VIX (with fallback)
    vix = get_current_vix()
    if vix is None:
        vix = 18.0  # conservative fallback

    # 3. Time-of-Day Bucket
    time_bucket = get_time_of_day_bucket()

    # 4. Trade Type Encoding
    trade_type_val = 0 if trade_type.lower() == "day" else 1

    # 5. Volatility Estimate (normalized from VIX)
    volatility = round(vix / 100.0, 4)

    return [streak, vix, time_bucket, trade_type_val, volatility]