from strategy import generate_trade_signal
from utils.logger import bot_logger  # Swapped from log_info to bot_logger for consistency
from telegram_bot import send_telegram_message
import traceback

def evaluate_entry_signals(market_data, indicators, sentiment, confidence_score):
    """
    Evaluates whether to enter a CALL or PUT trade based on strategy output.

    :param market_data: dict of market prices, etc.
    :param indicators: dict of technical indicator values
    :param sentiment: float from NLP/sentiment module
    :param confidence_score: float from ML model
    :return: "CALL", "PUT", or None
    """
    try:
        signal = generate_trade_signal(market_data, indicators, sentiment, confidence_score)

        if signal == "buy_call":
            bot_logger.info("📥 Entry Signal: BUY CALL triggered.")
            return "CALL"
        elif signal == "buy_put":
            bot_logger.info("📥 Entry Signal: BUY PUT triggered.")
            return "PUT"
        else:
            bot_logger.info("🔍 Entry Signal: No valid signal generated.")
            return None

    except Exception as e:
        bot_logger.error(f"[Entry Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Entry Module Error*\n"
            f"Could not evaluate entry signal.\n"
            f"Reason: `{str(e)}`"
        )
        return None  # Safe fallback