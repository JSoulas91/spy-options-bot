import logging
from datetime import datetime

# === Logger Setup ===
def setup_logger(name, log_file, level=logging.INFO):
    """Set up a custom logger with file output."""
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# === Price Rounding ===
def round_price(price, tick_size=0.05):
    """Round a price to the nearest tick size."""
    return round(round(price / tick_size) * tick_size, 2)

# === Timestamp Formatter ===
def format_timestamp(ts=None):
    """Return formatted timestamp string."""
    if ts is None:
        ts = datetime.utcnow()
    return ts.strftime('%Y-%m-%d %H:%M:%S')

# === Percentage Change ===
def calculate_pct_change(new_value, old_value):
    """Calculate percentage change."""
    try:
        return round(((new_value - old_value) / old_value) * 100, 2)
    except ZeroDivisionError:
        return 0.0
