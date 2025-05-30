# trade_manager.py

import datetime
import pytz
from config import MAX_RETRIES, SWING_TRADE_CONFIDENCE_THRESHOLD, VIX_SAFE_FOR_SWING
from utils.logger import bot_logger
from utils.vix_utils import fetch_vix_price, should_throttle_trades, is_vix_moderately_high
from strategy.event_filter import has_major_event_on
from trading.broker import execute_order, close_position

eastern = pytz.timezone("US/Eastern")

def should_convert_to_weekend_swing(confidence, vix_value, setup_valid):
    """
    Evaluates if a Friday trade should be held over the weekend.
    Conditions:
    - High confidence
    - VIX low
    - No major Monday event
    - Setup is valid
    """
    now = datetime.datetime.now(eastern)
    if now.weekday() != 4:  # Not Friday
        return False

    monday = now.date() + datetime.timedelta(days=3)

    if vix_value is None or vix_value > VIX_SAFE_FOR_SWING:
        bot_logger.info(f"⛔️ VIX too high for weekend swing: {vix_value}")
        return False

    if confidence < SWING_TRADE_CONFIDENCE_THRESHOLD:
        bot_logger.info(f"❌ Confidence too low for weekend swing: {confidence}")
        return False

    if not setup_valid:
        bot_logger.info("📉 Technical setup invalid for weekend hold.")
        return False

    if has_major_event_on(monday):
        bot_logger.info("📅 Major Monday event blocks weekend swing.")
        return False

    bot_logger.info("✅ Holding trade over weekend — all conditions met.")
    return True


def safe_execute_trade(trade_signal):
    """
    Attempt to execute a trade with retry logic.
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            order = execute_order(trade_signal)
            bot_logger.info(f"✅ Trade executed on attempt {retry_count + 1}")
            return order
        except Exception as e:
            bot_logger.warning(f"⚠️ Trade execution failed: {e}")
            retry_count += 1
    bot_logger.error("❌ Trade execution failed after maximum retries.")
    return None


def manage_exit(position, exit_signal):
    """
    Safely close a position based on exit signal or end-of-day logic.
    """
    try:
        result = close_position(position, exit_signal)
        bot_logger.info(f"💼 Position closed: {result}")
        return result
    except Exception as e:
        bot_logger.error(f"❌ Failed to close position: {e}")
        return None