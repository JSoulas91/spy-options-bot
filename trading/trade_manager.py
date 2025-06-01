# trade_manager.py

import time
from datetime import datetime, timedelta
from utils.logger import bot_logger
from utils.telegram_notifier import TelegramNotifier
from utils.vix_utils import get_vix_level
from utils.trade_tracker import TradeTracker
from strategy.event_filter import is_blackout_day
import config

telegram = TelegramNotifier()
tracker = TradeTracker()


def get_days_to_expiry(contract):
    try:
        expiry_str = contract.get("expiry")
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        return (expiry_date - today).days
    except Exception as e:
        bot_logger.warning(f"[Expiry Parse Error] {e}")
        return 0  # Assume expired if invalid


def execute_trade_with_retries(trade_function, contract):
    """Attempts trade with retries."""
    for attempt in range(1, config.MAX_RETRIES_PER_TRADE + 1):
        try:
            trade_function(contract)
            bot_logger.info(f"✅ Trade executed on attempt {attempt}")
            return True
        except Exception as e:
            bot_logger.warning(f"⚠️ Trade attempt {attempt} failed: {e}")
            if attempt < config.MAX_RETRIES_PER_TRADE:
                time.sleep(config.RETRY_DELAY_SECONDS)
    bot_logger.error("❌ Trade failed after all retry attempts.")
    return False


def evaluate_weekend_swing_hold(contract):
    """Determines if a Friday trade should be held as a weekend swing."""
    try:
        now = datetime.now()
        is_friday = now.weekday() == 4

        if not is_friday:
            return False

        confidence = contract.get("confidence", 0)
        if confidence < config.SWING_TRADE_THRESHOLD:
            bot_logger.info("❌ Swing rejected: confidence too low.")
            return False

        if config.ENABLE_VIX_THROTTLING:
            vix = get_vix_level()
            if vix is None:
                bot_logger.warning("⚠️ Could not retrieve VIX — rejecting swing hold.")
                return False
            if vix > config.VIX_MAX_THRESHOLD:
                bot_logger.info(f"❌ Swing rejected: VIX too high ({vix}).")
                return False

        if config.ENABLE_EVENT_BLACKOUT:
            monday = now + timedelta(days=3)
            if is_blackout_day(monday):
                bot_logger.info("❌ Swing rejected: Monday has blackout event.")
                return False

        telegram.send_swing_hold_alert(contract, reason="High confidence + low VIX + no Monday risk")
        bot_logger.info("✅ Holding position over weekend as swing.")
        return True

    except Exception as e:
        bot_logger.warning(f"[Swing Evaluation Error] {e}")
        return False


def manage_trade_execution(trade_function, contract):
    """Handles trade execution, expiry checks, PDT limits, and swing evaluation."""
    try:
        # PDT restriction
        if not tracker.can_execute_trade():
            bot_logger.info("🚫 Trade blocked due to PDT day trade limit.")
            return False

        # Expiry protection
        dte = get_days_to_expiry(contract)
        if dte <= 0:
            bot_logger.warning("⚠️ Trade skipped: Contract already expired or expires today.")
            telegram.send_message(f"⛔ Trade skipped — contract expires too soon (DTE={dte}).")
            return False

        trade_success = execute_trade_with_retries(trade_function, contract)

        if trade_success:
            tracker.increment_trade_count()

            # Evaluate for possible weekend swing
            contract["swing_held"] = evaluate_weekend_swing_hold(contract)

        return trade_success

    except Exception as e:
        bot_logger.error(f"[Trade Manager Error] {e}")
        return False