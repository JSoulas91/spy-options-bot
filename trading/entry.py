from strategy import generate_trade_signal
from utils.logger import bot_logger
from telegram_bot import send_telegram_message
import traceback

def evaluate_entry_signals(market_data, indicators, sentiment, confidence_score):
    """
    Determines whether to enter a CALL or PUT trade based on strategy signal + confidence.
    Returns: "CALL", "PUT", or None
    """
    try:
        signal = generate_trade_signal(market_data, indicators, sentiment, confidence_score)

        if signal == "buy_call" and confidence_score >= 0.5:
            bot_logger.info(f"📥 Entry: CALL (confidence: {confidence_score:.2f})")
            return "CALL"
        elif signal == "buy_put" and confidence_score >= 0.5:
            bot_logger.info(f"📥 Entry: PUT (confidence: {confidence_score:.2f})")
            return "PUT"
        else:
            bot_logger.info(f"🔍 No entry — Signal: {signal}, Confidence: {confidence_score:.2f}")
            return None

    except Exception as e:
        bot_logger.error(f"[Entry Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Entry Module Error*\nCould not evaluate entry.\nReason: `{str(e)}`"
        )
        return None