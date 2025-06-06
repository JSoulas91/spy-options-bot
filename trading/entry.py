# entry.py
import time
from datetime import datetime
from utils.logger        import bot_logger
from utils.trade_tracker import TradeTracker
from utils.trade_logger  import log_trade
from meta.meta_agent     import meta_agent
from meta.meta_state     import build_meta_state_for_entry
from trade_manager       import execute_trade_with_retries
from strategy            import evaluate_trade_signal
from utils.telegram_notifier import TelegramNotifier

trade_tracker = TradeTracker()
notifier      = TelegramNotifier()

def handle_entry(market_data):
    try:
        now = datetime.now()

        # 🚫 Block entries after 3:30 PM ET
        if now.hour == 15 and now.minute >= 30:
            bot_logger.info("[Entry] Blocked new entries after 3:30 PM ET.")
            return

        # 🔍 Evaluate trade signal
        signal = evaluate_trade_signal(market_data)
        if not signal.get("should_trade"):
            return

        confidence  = signal["confidence"]
        trade_type  = signal["trade_type"]      # 0 = day, 1 = swing
        trade_setup = signal["trade_setup"]

        option_symbol = trade_setup.get("option_symbol")

        # 🧠 Build meta‑state with live SPY + option quote (cached inside build_meta_state_for_entry)
        meta_state = build_meta_state_for_entry(
            data_1m   = signal["tf_1m"],
            data_5m   = signal["tf_5m"],
            data_15m  = signal["tf_15m"],
            data_1h   = signal["tf_1h"],
            data_1d   = signal["tf_1d"],
            confidence_score = confidence,
            trade_type       = trade_type,
            past_trades      = trade_tracker.get_open_trades(),
            long_term_data   = signal.get("long_term_data", {}),
            option_symbol    = option_symbol  # Pass option symbol for live quote
        )

        try:
            meta_agent.eval_mode()
            action = meta_agent.select_action(meta_state)
        except Exception as e:
            bot_logger.error(f"[Entry] Meta‑agent failed: {e}")
            return

        if action == 0:
            bot_logger.info("[Entry] Meta‑agent rejected trade setup.")
            return

        # 🚦 Max‑open‑trades gate
        if not trade_tracker.can_place_trade():
            bot_logger.info("[Entry] Blocked: Max open trades limit reached.")
            return

        # ✅ Execute trade via Tradier (w/ retries)
        order = execute_trade_with_retries(trade_setup)
        if not order:
            bot_logger.warning("[Entry] Trade execution failed after retries.")
            return

        # 📦 Enrich order metadata
        order.update({
            "confidence": confidence,
            "indicators": signal.get("indicators", {}),
            "timestamp":  now.isoformat(),
            "trade_type": trade_type,
            "meta_state": meta_state.tolist(),   # serialize as list for JSON
            "meta_action": int(action),
            "option_symbol": option_symbol        # store for exit meta‑state
        })

        # 🧾 Track + persist
        trade_tracker.log_trade(order, trade_type)
        log_trade({
            "timestamp": now.isoformat(),
            "action":    "buy",
            "trade_type": trade_type,
            "symbol":     order.get("symbol", "UNKNOWN"),
            "confidence_score": confidence,
            "trade_id":   order.get("id", "N/A")
        })

        # 📢 Telegram
        notifier.send_message(
            f"🟢 New {'Day' if trade_type == 0 else 'Swing'} Trade Executed: "
            f"{order.get('symbol', 'UNKNOWN')} (Conf {confidence:.2f})"
        )
        bot_logger.info(f"[Entry] {'Day' if trade_type==0 else 'Swing'} trade executed.")

    except Exception as e:
        bot_logger.exception(f"[Entry] Unhandled error: {e}")