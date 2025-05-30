import traceback
from strategy import generate_trade_signal
from event_filter import is_blackout_time, is_fed_event_today
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message

def evaluate_entry_signals(market_data, indicators, sentiment, confidence_score):
    """
    Determines whether to enter a CALL or PUT trade based on strategy signal, sentiment, indicators,
    confidence, and macro filters (economic blackouts, Fed speeches).
    Returns: "CALL", "PUT", or None
    """
    try:
        # Check for economic event blackout
        in_blackout, blackout_event = is_blackout_time()
        if in_blackout:
            logger.warning(f"🚫 Entry blocked due to economic blackout: {blackout_event}")
            send_telegram_message(
                f"🚫 *Entry Blocked — Economic Event*\n"
                f"Event: `{blackout_event}`\n"
                f"Skipping trade to reduce risk."
            )
            return None

        # Check for Fed speech or announcement risk
        fed_event, fed_event_name = is_fed_event_today()
        if fed_event:
            logger.warning(f"📢 Caution: Fed-related event today — {fed_event_name}")
            send_telegram_message(
                f"📢 *Caution — Fed Event Today*\n"
                f"Event: `{fed_event_name}`\n"
                f"Proceeding with extra caution."
            )

        signal = generate_trade_signal(market_data, indicators, sentiment, confidence_score)

        if signal == "buy_call" and confidence_score >= 0.5:
            logger.info(f"📥 Entry: CALL (Confidence: {confidence_score:.2f})")
            return "CALL"
        elif signal == "buy_put" and confidence_score >= 0.5:
            logger.info(f"📥 Entry: PUT (Confidence: {confidence_score:.2f})")
            return "PUT"
        else:
            logger.info(f"🔍 No entry — Signal: {signal}, Confidence: {confidence_score:.2f}")
            return None

    except Exception as e:
        logger.error(f"[Entry Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Entry Module Error*\nCould not evaluate entry.\nReason: `{str(e)}`"
        )
        return None