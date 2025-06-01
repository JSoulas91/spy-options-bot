import pytz
from datetime import datetime
from utils.trade_tracker import trade_tracker
from utils.trade_logger import log_trade_exit
from utils.logger import bot_logger
from trade_manager import close_trade
from meta.meta_agent import evaluate_exit_decision
from utils.telegram import send_telegram_message  # Make sure this exists

# Eastern time setup
eastern = pytz.timezone('US/Eastern')

def handle_exit():
    try:
        now = datetime.now(eastern)

        # === ⏰ 3:55 PM ET forced exit for day trades only ===
        if now.hour == 15 and now.minute >= 55:
            closed_ids = []
            for trade in trade_tracker.get_open_trades():
                if trade.get("trade_type") == 0:  # 0 = day trade
                    close_trade(trade)
                    trade_tracker.mark_trade_closed(trade["id"])
                    log_trade_exit(trade)
                    closed_ids.append(trade["id"])
                    bot_logger.info(
                        f"[TIME-BASED EXIT] Closed day trade {trade['id']} at 3:55 PM ET."
                    )

            # Send Telegram alert
            if closed_ids:
                message = (
                    f"📉 [Auto Exit @ 3:55 PM ET]\n"
                    f"Closed {len(closed_ids)} open day trade(s): {', '.join(map(str, closed_ids))}"
                )
                send_telegram_message(message)

            return  # Skip further checks this tick

        # === Smart exit logic for all trades (day or swing) ===
        for trade in trade_tracker.get_open_trades():
            exit_reason = should_exit_trade(trade)
            if exit_reason:
                close_trade(trade)
                trade_tracker.mark_trade_closed(trade["id"])
                log_trade_exit(trade)
                bot_logger.info(
                    f"[EXIT] Closed trade {trade['id']} due to: {exit_reason}"
                )
                send_telegram_message(
                    f"🚪 Exited trade {trade['id']} early due to: {exit_reason}"
                )

    except Exception as e:
        bot_logger.error(f"[EXIT ERROR] Failed to handle exits: {str(e)}")

def should_exit_trade(trade):
    """
    Determines whether the trade should be closed early using meta-agent + indicators.
    Returns a reason string if it should exit, or None otherwise.
    """
    try:
        # Evaluate with meta-agent logic
        should_exit = evaluate_exit_decision(trade)
        if should_exit:
            return "Meta-agent signal"

        # TODO: Add more rules like RSI, MACD, ATR-based stop, etc. if needed
        return None  # Hold

    except Exception as e:
        bot_logger.error(f"[should_exit_trade] Error evaluating exit: {str(e)}")
        return None