import pytz
from datetime import datetime
from utils.trade_tracker import trade_tracker
from utils.trade_logger import log_trade_exit
from utils.logger import bot_logger
from trade_manager import close_trade
from meta.meta_agent import evaluate_exit_decision
from utils.telegram_notifier import TelegramNotifier

# Eastern time zone
eastern = pytz.timezone('US/Eastern')
notifier = TelegramNotifier()  # Initialize once

def handle_exit():
    try:
        now = datetime.now(eastern)

        # 🕒 Force close all open day trades at 3:55 PM ET
        if now.hour == 15 and now.minute >= 55:
            closed_ids = []
            for trade in trade_tracker.get_open_trades():
                if trade.get("trade_type") == 0:  # 0 = day trade
                    close_trade(trade)
                    trade_tracker.mark_trade_closed(trade["id"])
                    log_trade_exit(trade)
                    closed_ids.append(trade["id"])
                    bot_logger.info(f"[TIME EXIT] Closed day trade {trade['id']} at 3:55 PM ET")

            if closed_ids:
                notifier.send_message(
                    f"📉 [Auto Exit @ 3:55 PM ET]\nClosed {len(closed_ids)} day trade(s): {', '.join(map(str, closed_ids))}"
                )
            return  # Skip rest if it's time-based closure

        # 🤖 Smart exit logic for both day and swing trades
        for trade in trade_tracker.get_open_trades():
            exit_reason = should_exit_trade(trade)
            if exit_reason:
                close_trade(trade)
                trade_tracker.mark_trade_closed(trade["id"])
                log_trade_exit(trade)
                bot_logger.info(f"[EXIT] Closed trade {trade['id']} due to: {exit_reason}")
                notifier.send_message(f"🚪 Exited trade {trade['id']} due to: {exit_reason}")

    except Exception as e:
        bot_logger.error(f"[EXIT ERROR] Failed to handle exits: {str(e)}")

def should_exit_trade(trade):
    """
    Decide if the trade should be closed early (non-time-based exit).
    """
    try:
        if evaluate_exit_decision(trade):
            return "Meta-agent signal"
        # (Optional: Add more custom indicator logic here)
        return None
    except Exception as e:
        bot_logger.error(f"[Exit Evaluation Error] {str(e)}")
        return None