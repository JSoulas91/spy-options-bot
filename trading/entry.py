# entry.py

import time
from datetime import datetime
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

        # 🚫 Block entries after 3:30 PM ET
        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("[Entry] Blocked new entries after 3:30 PM ET.")
            return

        # 🔍 Evaluate trade signal
        signal = evaluate_trade_signal(market_data)
        if not signal["should_trade"]:
            return

        confidence = signal["confidence"]
        trade_type = signal["trade_type"]  # 0 = day, 1 = swing
        trade_setup = signal["trade_setup"]

        # 🧠 Meta-agent decision
        meta_state = build_meta_state_for_entry(
            market_data,
            confidence_score=confidence,
            trade_type=trade_type
        )
        meta_agent.eval_mode()
        action = meta_agent.select_action(meta_state)

        if action == 0:
            bot_logger.info("[Entry] Meta-agent rejected trade setup.")
            return

        # 🚦 Check max open trades limit before placing new trade
        if not trade_tracker.can_place_trade():
            bot_logger.info("[Entry] Blocked: Max open trades limit reached. Skipping new trade.")
            return

        # ✅ Execute trade via Tradier with retries
        order = execute_trade_with_retries(trade_setup)

        if not order:
            bot_logger.warning("[Entry] Trade execution failed after retries.")
            return

        # 📦 Enrich metadata
        order.update({
            "confidence": confidence,
            "indicators": signal.get("indicators", {}),
            "timestamp": now.isoformat(),
            "trade_type": trade_type,
            "meta_state": meta_state,
            "meta_action": action
        })

        # 🧾 Log and track trade
        trade_tracker.log_trade(order, trade_type)
        log_trade({
            "timestamp": now.isoformat(),
            "action": "buy",
            "trade_type": trade_type,
            "symbol": order.get("symbol", "UNKNOWN"),
            "confidence_score": confidence,
            "trade_id": order.get("id", "N/A"),
            "indicators": str(order.get("indicators", {}))
        })

        # 📢 Notify
        notifier.send_message(
            f"🟢 New {'Day' if trade_type == 0 else 'Swing'} Trade Executed: {order.get('symbol', 'UNKNOWN')} "
            f"(Confidence: {confidence:.2f})"
        )
        bot_logger.info(f"[Entry] {'Day' if trade_type == 0 else 'Swing'} trade executed and logged.")

    except Exception as e:
        bot_logger.error(f"[Entry] Error: {str(e)}")