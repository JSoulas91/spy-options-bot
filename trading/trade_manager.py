import time
import traceback
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from entry import evaluate_entry_signals
from exit import check_exit_conditions
from config import DEFAULT_POSITION_SIZE, MAX_RETRIES_PER_TRADE, RETRY_DELAY_SECONDS

active_trades = []

def execute_trade(order):
    """
    Simulates trade execution logic.
    Replace this with your broker API (e.g., Alpaca) execution logic.
    """
    try:
        # Example simulation: pretend it fails randomly
        # Replace this with actual broker call
        success = True  # Set False to test retry
        if not success:
            raise Exception("Simulated broker failure")
        return True
    except Exception as e:
        logger.warning(f"[Trade Execution] Failed to execute trade: {e}")
        return False

def manage_trades(market_data):
    global active_trades
    try:
        for trade in list(active_trades):
            try:
                action = check_exit_conditions(trade, market_data)
                if action == "exit":
                    logger.info(f"🚪 Exiting trade: {trade}")
                    active_trades.remove(trade)
                    send_telegram_message(f"✅ *Trade Exited*\n{trade}")
            except Exception as e:
                logger.error(f"[Exit Evaluation Error] {str(e)}")
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
            success = execute_trade(order)
            if success:
                logger.info("✅ Trade executed successfully.")
                send_telegram_message(f"✅ *Trade Executed Successfully*\n{order}")
                break
            else:
                logger.warning("⚠️ Trade execution failed.")
                time.sleep(RETRY_DELAY_SECONDS)

        if not success:
            logger.error("🚫 Max trade retries exceeded. Aborting trade.")
            send_telegram_message(f"🚫 *Trade Failed After {MAX_RETRIES_PER_TRADE} Attempts*\n{order}")
            return

        # Only store the trade if successful
        active_trades.append(order)

    except Exception as e:
        logger.error(f"[Trade Entry Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ *Trade Entry Error*\n{e}")