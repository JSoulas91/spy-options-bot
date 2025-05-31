# entry.py
import traceback
from strategy import generate_trade_signal
from meta.meta_agent import MetaAgent
from meta.meta_state import build_meta_state_for_entry
from utils.logger import bot_logger
from telegram_bot import send_telegram_message

meta_agent = MetaAgent()
meta_agent.load_model()

def evaluate_entry_signals(market_data, indicators, sentiment, confidence_score):
    """
    Determines whether to enter a CALL or PUT trade based on strategy signal, confidence, and meta-agent action.
    Returns: "CALL", "PUT", or None
    """
    try:
        signal = generate_trade_signal(market_data, indicators, sentiment, confidence_score)
        if signal not in ["buy_call", "buy_put"]:
            bot_logger.info(f"🔍 No entry — Signal: {signal}")
            return None

        # Build meta-state and check meta-agent decision
        meta_state = build_meta_state_for_entry(market_data, confidence_score)
        meta_action = meta_agent.select_action(meta_state)

        if meta_action == 0:
            bot_logger.info(f"🧠 Meta-Agent blocked trade — Action: {meta_action}")
            return None

        direction = "CALL" if signal == "buy_call" else "PUT"
        bot_logger.info(f"📥 Entry: {direction} (Confidence: {confidence_score:.2f} | Meta: {meta_action})")
        return direction

    except Exception as e:
        bot_logger.error(f"[Entry Evaluation Error] {str(e)}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Entry Evaluation Error*\n```{str(e)}```"
        )
        return None