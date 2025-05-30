import time
import traceback
from datetime import datetime, timedelta

from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from entry import evaluate_entry_signals
from exit import check_exit_conditions
from config import (
    DEFAULT_POSITION_SIZE,
    MAX_RETRIES_PER_TRADE,
    RETRY_DELAY_SECONDS,
)
from utils.vix_utils import fetch_vix_price
from utils.calendar import has_major_event_on  # Assuming this exists for economic events

active_trades = []

def execute_trade(order):
    """
    Simulates trade execution logic.
    Replace this with your broker API (e.g., Alpaca) execution logic.
    """
    try:
        success = True  # Set False to simulate failure for testing
        if not success:
            raise Exception("Simulated broker failure")
        return True
    except Exception as e:
        logger.warning(f"[Trade Execution] Failed to execute trade: {e}")
        return False

def is_fatal_error(error: Exception) -> bool:
    """
    Checks if the error is fatal and should not be retried.
    """
    fatal_errors = [
        "invalid symbol", "bad request", "insufficient funds", "permission denied"
    ]
    return any(msg in str(error).lower() for msg in fatal_errors)

def is_viable_weekend_swing(order, confidence_score):
    """
    Determines if a trade is safe to hold over the weekend.
    Checks VIX, confidence, and Monday's economic calendar.
    """
    vix = fetch_vix_price()
    monday = (datetime.now() + timedelta(days=3)).date()

    reasons = []

    if order["type"] != "swing":
        reasons.append("Trade is not a swing type.")
    if vix is None:
        reasons.append("Unable to fetch VIX data.")
    elif vix > 22:
        reasons.append(f"VIX too high: {vix}")
    if confidence_score < 0.85:
        reasons.append(f"Confidence score too low: {confidence_score}")
    if has_major_event_on(monday):
        reasons.append("Monday has high-impact economic events.")

    if reasons:
        logger.info(f"Weekend swing rejected: {reasons}")
        return False, reasons
    return True, ["Favorable VIX", "High confidence", "No major events Monday"]

def manage_trades(market_data):
    global active_trades
    try:
        for trade in list(active_trades):
            try:
                # Check if we should exit the trade
                action = check_exit_conditions(trade, market_data)
                if action == "exit":
                    logger.info(f"🚪 Exiting trade: {trade}")
                    active_trades.remove(trade)
                    send_telegram_message(f"✅ *Trade Exited*\n{trade}")
                else:
                    # Weekend swing logic: if it is Friday afternoon, decide to hold or exit swings
                    now = datetime.now()
                    if trade["type"] == "swing" and now.weekday() == 4:  # Friday
                        confidence_score = market_data.get("confidence_score", 1.0)
                        viable, reasons = is_viable_weekend_swing(trade, confidence_score)
                        if not viable:
                            logger.info(f"🚪 Exiting weekend swing trade due to: {reasons}")
                            active_trades.remove(trade)
                            send_telegram_message(
                                f"✅ *Weekend Swing Trade Closed*\nReason: {reasons}\n{trade}"
                            )
                        else:
                            logger.info(f"⏳ Holding weekend swing trade: {trade}")
                            send_telegram_message(
                                f"⏳ *Holding Weekend Swing Trade*\nReasons: {reasons}\n{trade}"
                            )

            except Exception as e:
                logger.error(f"[Exit Evaluation Error] {str(e)}")
                logger.debug(traceback.format_exc())
                send_telegram_message(f"⚠️ *Exit Evaluation Error*\n{e}")

    except Exception as e:
        logger.error(f"[Trade Manager Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ *Trade Manager Error*\n{str(e)}")

def try_trade_entry(market_data, indicators, sentiment, confidence_score):
    """
    Attempts to enter a trade with retry logic on failure.
    """
    try:
        direction = evaluate_entry_signals(market_data, indicators, sentiment, confidence_score)
        if not direction:
            return

        symbol = market_data.get("symbol", "SPY")
        entry_price = market_data.get("price")

        order = {
            "symbol": symbol,
            "direction": direction,
            "position_size": DEFAULT_POSITION_SIZE,
            "entry_price": entry_price,
            "timestamp": market_data.get("timestamp"),
            "type": "day" if market_data.get("is_day_trade") else "swing",
        }

        success = False

        for attempt in range(1, MAX_RETRIES_PER_TRADE + 1):
            logger.info(f"⚙️ Attempt {attempt}/{MAX_RETRIES_PER_TRADE} to execute trade: {order}")
            send_telegram_message(f"🔁 *Trade Execution Attempt {attempt}*\n{order}")

            try:
                success = execute_trade(order)
                if success:
                    logger.info("✅ Trade executed successfully.")
                    send_telegram_message(f"✅ *Trade Executed Successfully*\n{order}")
                    break
            except Exception as e:
                if is_fatal_error(e):
                    logger.error(f"❌ Fatal trade execution error: {str(e)} — aborting retries.")
                    send_telegram_message(f"❌ *Fatal Trade Error — Aborting*\n{str(e)}")
                    return
                else:
                    logger.warning(f"⚠️ Retryable trade error: {str(e)} — will retry.")
                    send_telegram_message(f"⚠️ *Trade Retry {attempt} Failed*\nReason: `{str(e)}`")

            time.sleep(RETRY_DELAY_SECONDS)

        if not success:
            logger.error("🚫 Max trade retries exceeded. Aborting trade.")
            send_telegram_message(f"🚫 *Trade Failed After {MAX_RETRIES_PER_TRADE} Attempts*\n{order}")
            return

        # Only store successful trades
        active_trades.append(order)

    except Exception as e:
        logger.error(f"[Trade Entry Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ *Trade Entry Error*\n{e}")