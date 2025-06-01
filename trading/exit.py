from datetime import datetime
from utils.logger import bot_logger
from utils.trade_tracker import TradeTracker
from utils.trade_logger import log_trade
from trade_manager import close_trade
from meta.meta_agent import meta_agent
from meta.meta_state import build_meta_state_for_exit
from strategy import evaluate_exit_signal

trade_tracker = TradeTracker()

def handle_exit(market_data):
    try:
        now = datetime.now()

        # Time-based forced exit for day trades only (3:55 PM ET)
        if now.hour == 15 and now.minute >= 55:
            for trade in trade_tracker.get_open_trades():
                if trade["trade_type"] == 0:  # Only force-close day trades
                    close_trade(trade)
                    trade_tracker.mark_trade_closed(trade["id"])

                    log_trade({
                        "timestamp": now.isoformat(),
                        "action": "sell",
                        "trade_type": trade["trade_type"],
                        "symbol": trade["symbol"],
                        "profit_loss": trade["profit"],
                        "trade_duration": (now - trade["entry_time"]).total_seconds() / 60,
                        "confidence_score": trade.get("confidence", 0),
                        "indicators": str(trade.get("indicators", {}))
                    })

                    bot_logger.info(f"[Exit] Forced day trade exit for {trade['id']} at 3:55 PM.")
            return

        # Meta-agent guided exit logic
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

                log_trade({
                    "timestamp": now.isoformat(),
                    "action": "sell",
                    "trade_type": trade["trade_type"],
                    "symbol": trade["symbol"],
                    "profit_loss": profit,
                    "trade_duration": trade_duration,
                    "confidence_score": confidence,
                    "indicators": str(trade.get("indicators", {}))
                })

                bot_logger.info(f"[Exit] Meta-agent exited trade {trade['id']} at profit: {profit:.2%}")

    except Exception as e:
        bot_logger.error(f"[Exit] Error: {str(e)}")