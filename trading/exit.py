# exit.py

import pytz
from datetime import datetime
from utils.trade_tracker import trade_tracker
from utils.trade_logger import log_trade_exit
from utils.logger import bot_logger
from trade_manager import close_trade
from meta.meta_agent import evaluate_exit_decision
from utils.telegram_notifier import TelegramNotifier

eastern = pytz.timezone('US/Eastern')
notifier = TelegramNotifier()

def get_days_to_expiry(contract):
    try:
        expiry_str = contract.get("expiry")
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = datetime.now(eastern).date()
        return (expiry_date - today).days
    except Exception as e:
        bot_logger.warning(f"[Expiry Parse Error] {e}")
        return 0

def handle_exit():
    try:
        now = datetime.now(eastern)

        # 🕒 Force close all day trades at 3:55 PM
        if now.hour == 15 and now.minute >= 55:
            closed_ids = []
            for trade in trade_tracker.get_open_trades():
                if trade.get("trade_type") == 0:
                    close_trade(trade)
                    trade_tracker.mark_trade_closed(trade.get("id"))
                    log_trade_exit(trade)
                    closed_ids.append(str(trade.get("id", '?')))
                    bot_logger.info(f"[TIME EXIT] Closed day trade {trade.get('id')} at 3:55 PM ET")

            if closed_ids:
                notifier.send_message(
                    f"📉 [Auto Exit @ 3:55 PM ET]\nClosed {len(closed_ids)} day trade(s): {', '.join(closed_ids)}"
                )
            return

        # 🔍 Evaluate exits for all trades
        for trade in trade_tracker.get_open_trades():
            exit_reason = should_exit_trade(trade)
            if exit_reason:
                close_trade(trade)
                trade_tracker.mark_trade_closed(trade.get("id"))
                log_trade_exit(trade)
                bot_logger.info(f"[EXIT] Closed trade {trade.get('id')} due to: {exit_reason}")
                notifier.send_message(f"🚪 Exited trade {trade.get('id')} due to: {exit_reason}")

    except Exception as e:
        bot_logger.error(f"[EXIT ERROR] Failed to handle exits: {str(e)}")

def should_exit_trade(trade):
    try:
        if evaluate_exit_decision(trade):
            return "Meta-agent signal"

        if trade.get("trade_type") == 1:
            dte = get_days_to_expiry(trade)
            if dte <= 1:
                return f"Contract near expiry (DTE={dte})"

        return None

    except Exception as e:
        bot_logger.error(f"[Exit Evaluation Error] {str(e)}")
        return None