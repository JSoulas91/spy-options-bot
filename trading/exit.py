# exit.py

import traceback
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from event_filter import is_blackout_time
from utils.trade_tracker import close_trade  # 🔹 NEW

def evaluate_exit_conditions(
    position,
    market_data,
    indicators,
    confidence_score,
    trailing_stop_enabled=True,
):
    try:
        exit_signal = False
        reason = ""

        price = market_data.get("price")
        entry_price = position.get("entry_price")
        atr = indicators.get("atr")
        trailing_stop_hit = position.get("trailing_stop_hit", False)

        blackout, event_name = is_blackout_time()
        if blackout:
            logger.warning(f"🚫 Economic Event Blackout — Exiting due to {event_name}")
            exit_signal = True
            reason = f"Economic Blackout: {event_name}"

        if confidence_score < 0.3:
            exit_signal = True
            reason = "📉 Extremely low confidence score"

        if trailing_stop_enabled and trailing_stop_hit:
            exit_signal = True
            reason = "🔻 Trailing stop triggered"

        if indicators.get("exit_signal", False):
            exit_signal = True
            reason = "⚠️ Technical exit signal"

        if atr and price and entry_price:
            stop_loss_atr = entry_price - (atr * 1.5)
            if price < stop_loss_atr:
                exit_signal = True
                reason = "🛑 ATR-based stop-loss triggered"

        if exit_signal:
            logger.info(f"🚪 Exit triggered — Reason: {reason}")
            send_telegram_message(f"🚪 *Exit Triggered*\nReason: {reason}")

            # 🔹 LOG THE CLOSED TRADE
            trade_id = position.get("id")
            if trade_id:
                close_trade(trade_id)

            return True

        return False

    except Exception as e:
        logger.error(f"[Exit Evaluation Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Exit Module Error*\nCould not evaluate exit.\nReason: `{str(e)}`"
        )
        return False