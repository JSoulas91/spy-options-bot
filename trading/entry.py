import time
from datetime import datetime
from config import MAX_DAY_TRADES, ENFORCE_PDT_LIMITS
from utils.logger import bot_logger
from utils.trade_tracker import TradeTracker
from meta.meta_agent import meta_agent
from meta.meta_state import build_meta_state_for_entry
from trade_manager import execute_trade_with_retries
from strategy import evaluate_trade_signal

trade_tracker = TradeTracker()

def handle_entry(market_data):
    try:
        now = datetime.now()
        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("Entry blocked after 3:30 PM ET.")
            return

        signal = evaluate_trade_signal(market_data)
        if not signal["should_trade"]:
            return

        confidence = signal["confidence"]
        trade_type = signal["trade_type"]  # 0 = day, 1 = swing

        if trade_type == 0:
            if ENFORCE_PDT_LIMITS and not trade_tracker.can_place_day_trade():
                bot_logger.info("Day trade entry blocked by PDT limits.")
                return
            if trade_tracker.get_today_day_trade_count() >= MAX_DAY_TRADES:
                bot_logger.info("Day trade entry blocked by max daily limit.")
                return

        meta_state = build_meta_state_for_entry(
            market_data,
            confidence_score=confidence,
            trade_type=trade_type
        )
        action = meta_agent.select_action(meta_state)

        if action == 0:
            bot_logger.info("Meta-agent declined entry signal.")
            return

        order_details = execute_trade_with_retries(signal["trade_setup"])
        if order_details:
            trade_tracker.log_trade(order_details, trade_type)
            bot_logger.info(f"{'Day' if trade_type == 0 else 'Swing'} trade executed and logged.")
        else:
            bot_logger.warning("Trade execution failed after retries.")

    except Exception as e:
        bot_logger.error(f"Entry error: {str(e)}")