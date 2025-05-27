import logging
from datetime import datetime
import pytz

# === Logger Setup ===
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# === Price Rounding ===
def round_price(price, tick_size=0.05):
    return round(round(price / tick_size) * tick_size, 2)

# === Timestamp Formatter ===
def format_timestamp(ts=None):
    if ts is None:
        ts = datetime.utcnow()
    return ts.strftime('%Y-%m-%d %H:%M:%S')

# === Percentage Change ===
def calculate_pct_change(new_value, old_value):
    try:
        return round(((new_value - old_value) / old_value) * 100, 2)
    except ZeroDivisionError:
        return 0.0

# === Trade Type Determination ===
eastern = pytz.timezone("US/Eastern")

def is_day_trade(position):
    """Returns True if trade was opened and still on the same calendar day (Eastern time)."""
    try:
        entry_time = datetime.fromisoformat(position['entry_time']).replace(tzinfo=pytz.utc).astimezone(eastern)
        now = datetime.now(eastern)
        return entry_time.date() == now.date()
    except Exception:
        return False

def is_swing_trade(position):
    return not is_day_trade(position)
