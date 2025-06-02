# entry.py

import time
from datetime import datetime
from config import MAX_DAY_TRADES, ENFORCE_PDT_LIMITS
from utils.logger import bot_logger
from utils.trade_tracker import TradeTracker
from utils.trade_logger import log_trade
from meta.meta_agent import meta_agent
from meta.meta_state import build_meta_state_for_entry
from trade_manager import execute_trade_with_retries
from strategy import evaluate_trade_signal
from utils.telegram_notifier import TelegramNotifier

trade_tracker = TradeTracker()
notifier = TelegramNotifier()

def handle_entry(market_data):
    try:
        now = datetime.now()

        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("[Entry] Blocked new entries after 3:30 PM ET.")
            return

        signal = evaluate_trade_signal(market_data)
        if not signal["should_trade"]:
            return

        confidence = signal["confidence"]
        trade_type = signal["trade_type"]

        if trade_type == 0:
            if ENFORCE_PDT_LIMITS and not trade_tracker.can_place_day_trade():
                bot_logger.info("[Entry] Blocked by PDT rules.")
                return
            if trade_tracker.get_today_day_trade_count() >= MAX_DAY_TRADES:
                bot_logger.info("[Entry] Blocked by max daily limit.")
                return

        meta_state = build_meta_state_for_entry(
            market_data,
            confidence_score=confidence,
            trade_type=trade_type
        )
        action = meta_agent.select_action(meta_state)

        if action == 0:
            bot_logger.info("[Entry] Meta-agent rejected trade setup.")
            return

        order_details = execute_trade_with_retries(signal["trade_setup"])
        if order_details:
            order_details["confidence"] = confidence
            order_details["indicators"] = signal.get("indicators", {})
            order_details["timestamp"] = now.isoformat()
            order_details["trade_type"] = trade_type
            order_details["meta_state"] = meta_state
            order_details["meta_action"] = [1.0, 0.0] if action == 1 else [0.0, 1.0]

            trade_tracker.log_trade(order_details, trade_type)

            log_trade({
                "timestamp": now.isoformat(),
                "action": "buy",
                "trade_type": trade_type,
                "symbol": order_details["symbol"],
                "confidence_score": confidence,
                "trade_id": order_details.get("id"),
                "indicators": str(order_details["indicators"])
            })

            notifier.send_message(f"🟢 New {'Day' if trade_type == 0 else 'Swing'} Trade Executed: {order_details['symbol']} (Confidence: {confidence:.2f})")

            bot_logger.info(f"[Entry] {'Day' if trade_type == 0 else 'Swing'} trade executed and logged.")
        else:
            bot_logger.warning("[Entry] Trade execution failed after retries.")

    except Exception as e:
        bot_logger.error(f"[Entry] Error: {str(e)}")