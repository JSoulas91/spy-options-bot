# trade_manager.py

import time
from datetime import datetime, timedelta
from utils.logger import bot_logger
from utils.telegram_notifier import TelegramNotifier
from utils.vix_utils import get_vix_level
from strategy.event_filter import is_blackout_day
from config import (
    MAX_RETRIES_PER_TRADE,
    RETRY_DELAY_SECONDS,
    USE_AGGRESSIVE_MODE,
    ENABLE_VIX_THROTTLING,
    ENABLE_EVENT_BLACKOUT,
    ENABLE_ADAPTIVE_CONFIDENCE,
    SWING_TRADE_THRESHOLD,
    VIX_MAX_THRESHOLD
)

telegram = TelegramNotifier()


def execute_trade_with_retries(trade_function, contract):
    """Attempts trade with retries."""
    for attempt in range(1, MAX_RETRIES_PER_TRADE + 1):
        try:
            trade_function(contract)
            bot_logger.info(f"✅ Trade executed on attempt {attempt}")
            return True
        except Exception as e:
            bot_logger.warning(f"⚠️ Trade attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES_PER_TRADE:
                time.sleep(RETRY_DELAY_SECONDS)
    bot_logger.error("❌ Trade failed after all retry attempts.")
    return False


def evaluate_weekend_swing_hold(contract):
    """
    Determines if a Friday trade should be held as a weekend swing.
    Returns True if safe to hold, else False.
    """
    try:
        now = datetime.now()
        is_friday = now.weekday() == 4  # Friday = 4

        if not is_friday:
            return False  # Only care on Friday

        confidence = contract.get("confidence", 0)
        if confidence < SWING_TRADE_THRESHOLD:
            bot_logger.info("❌ Swing rejected: confidence too low.")
            return False

        if ENABLE_VIX_THROTTLING:
            vix = get_vix_level()
            if vix > VIX_MAX_THRESHOLD:
                bot_logger.info(f"❌ Swing rejected: VIX too high ({vix}).")
                return False

        if ENABLE_EVENT_BLACKOUT:
            monday = now + timedelta(days=3)  # Check Monday
            if is_blackout_day(monday):
                bot_logger.info("❌ Swing rejected: Monday has blackout event.")
                return False

        # Passed all checks – allow swing
        telegram.send_swing_hold_alert(contract, reason="High confidence + low VIX + no Monday risk")
        bot_logger.info("✅ Holding position over weekend as swing.")
        return True

    except Exception as e:
        bot_logger.warning(f"[Swing Evaluation Error] {e}")
        return False


def manage_trade_execution(trade_function, contract):
    """
    Handles trade execution and weekend swing evaluation logic.
    """
    try:
        trade_success = execute_trade_with_retries(trade_function, contract)

        if trade_success:
            # Weekend swing logic only applies on Friday and for swing trades
            if evaluate_weekend_swing_hold(contract):
                contract["swing_held"] = True
            else:
                contract["swing_held"] = False

        return trade_success

    except Exception as e:
        bot_logger.error(f"[Trade Manager Error] {e}")
        return False