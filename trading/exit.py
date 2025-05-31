from datetime import datetime
from utils.logger import bot_logger
from utils.trade_tracker import TradeTracker
from trade_manager import close_trade
from meta.meta_agent import meta_agent
from meta.meta_state import build_meta_state_for_exit
from strategy import evaluate_exit_signal

trade_tracker = TradeTracker()

def handle_exit(market_data):
    try:
        now = datetime.now()

        # Close all open trades before 3:55 PM ET
        if now.hour == 15 and now.minute >= 55:
            for trade in trade_tracker.get_open_trades():
                close_trade(trade)
                trade_tracker.mark_trade_closed(trade["id"])
                bot_logger.info("Closed trade due to time-based exit (3:55 PM).")
            return

        for trade in trade_tracker.get_open_trades():
            trade_duration = (now - trade["entry_time"]).total_seconds() / 60
            result = evaluate_exit_signal(market_data, trade)
            confidence = result["confidence"]
            profit = result["current_profit"]

            meta_state = build_meta_state_for_exit(
                market_data,
                confidence_score=confidence,
                trade_type=trade["trade_type"],
                trade_duration_minutes=trade_duration,
                current_profit=profit
            )
            action = meta_agent.select_action(meta_state)

            if action == 1:
                close_trade(trade)
                trade_tracker.mark_trade_closed(trade["id"])
                bot_logger.info(f"Meta-agent exited trade {trade['id']} at profit: {profit:.2%}")

    except Exception as e:
        bot_logger.error(f"Exit error: {str(e)}")