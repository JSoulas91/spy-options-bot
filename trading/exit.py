from utils.logger import bot_logger
import traceback

def evaluate_exit_conditions(position, market_data, indicators, confidence_score, trailing_stop_enabled=True):
    """
    Determines whether to exit a position based on various conditions.
    
    :param position: dict of current position details
    :param market_data: dict of current price data
    :param indicators: dict of technical indicators
    :param confidence_score: float confidence level
    :param trailing_stop_enabled: bool
    :return: True if exit condition met, False otherwise
    """
    try:
        exit_signal = False
        reason = ""

        # Exit if confidence score is too low
        if confidence_score < 0.3:
            exit_signal = True
            reason = "📉 Low confidence score"

        # Exit if trailing stop was hit
        if trailing_stop_enabled and position.get("trailing_stop_hit", False):
            exit_signal = True
            reason = "🔻 Trailing stop hit"

        # Exit if a technical reversal signal is detected
        if indicators.get("exit_signal", False):
            exit_signal = True
            reason = "⚠️ Technical reversal signal"

        if exit_signal:
            bot_logger.info(f"🚪 Exit Signal Triggered — Reason: {reason}")
            return True

        return False

    except Exception as e:
        bot_logger.error(f"[Exit Evaluation Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        return False