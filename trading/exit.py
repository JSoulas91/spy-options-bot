from utils.logger import bot_logger
from telegram_bot import send_telegram_message
import traceback

def evaluate_exit_conditions(position, market_data, indicators, confidence_score, trailing_stop_enabled=True):
    """
    Determines whether to exit a position.
    Returns True if exit is required, False otherwise.
    """
    try:
        exit_signal = False
        reason = ""

        # Exit if confidence score is very low
        if confidence_score < 0.3:
            exit_signal = True
            reason = "📉 Low model confidence"

        # Trailing stop triggered
        if trailing_stop_enabled and position.get("trailing_stop_hit", False):
            exit_signal = True
            reason = "🔻 Trailing stop hit"

        # Technical exit signal (like EMA cross, MACD, RSI, etc.)
        if indicators.get("exit_signal", False):
            exit_signal = True
            reason = "⚠️ Technical reversal detected"

        # Time-based fallback exit — handled elsewhere but could be used here in future

        if exit_signal:
            bot_logger.info(f"🚪 Exit triggered — Reason: {reason}")
            return True

        return False

    except Exception as e:
        bot_logger.error(f"[Exit Evaluation Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Exit Module Error*\nCould not evaluate exit.\nReason: `{str(e)}`"
        )
        return False