import logging
from datetime import datetime
import pytz
import traceback
from utils.logger import bot_logger  # ✅ Use centralized logger

# === Logger Setup ===
def setup_logger(name, log_file, level=logging.INFO):
    try:
        formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger
    except Exception as e:
        bot_logger.error(f"[Logger Setup Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        return logging.getLogger(name)  # Return basic logger fallback

# === Price Rounding ===
def round_price(price, tick_size=0.05):
    try:
        return round(round(price / tick_size) * tick_size, 2)
    except Exception as e:
        bot_logger.error(f"[Price Rounding Error] {str(e)} | price={price}, tick_size={tick_size}")
        bot_logger.debug(traceback.format_exc())
        return price  # Fallback to unrounded price

# === Timestamp Formatter ===
def format_timestamp(ts=None):
    try:
        if ts is None:
            ts = datetime.utcnow()
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        bot_logger.error(f"[Timestamp Format Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        return "1970-01-01 00:00:00"  # Fallback epoch

# === Percentage Change ===
def calculate_pct_change(new_value, old_value):
    try:
        if old_value == 0:
            return 0.0
        return round(((new_value - old_value) / old_value) * 100, 2)
    except Exception as e:
        bot_logger.error(f"[Pct Change Error] {str(e)} | new={new_value}, old={old_value}")
        bot_logger.debug(traceback.format_exc())
        return 0.0

# === Trade Type Checkers ===
eastern = pytz.timezone("US/Eastern")

def is_day_trade(position):
    """Return True if entry_time is today (Eastern Time)."""
    try:
        entry_time = datetime.fromisoformat(position['entry_time']).astimezone(eastern)
        now = datetime.now(eastern)
        return entry_time.date() == now.date()
    except Exception as e:
        bot_logger.error(f"[is_day_trade Error] {str(e)} | position={position}")
        bot_logger.debug(traceback.format_exc())
        return False

def is_swing_trade(position):
    try:
        return not is_day_trade(position)
    except Exception as e:
        bot_logger.error(f"[is_swing_trade Error] {str(e)} | position={position}")
        bot_logger.debug(traceback.format_exc())
        return False