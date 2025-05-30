import pytz
import time
import traceback
from datetime import datetime
from alpaca_trade_api.rest import REST, TimeFrame

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, MAX_ORDER_RETRIES, RETRY_DELAY_SECONDS
from helpers import is_day_trade
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message

eastern = pytz.timezone("US/Eastern")
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')


def get_current_time_et():
    try:
        return datetime.now(eastern)
    except Exception as e:
        logger.error(f"[Time Conversion Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message("⚠️ Timezone conversion failed — fallback to UTC.")
        return datetime.utcnow()


def is_market_open():
    try:
        return alpaca.get_clock().is_open
    except Exception as e:
        logger.error(f"[Market Status Check Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message("⚠️ Could not check market status.")
        return False


def should_enter_day_trade():
    try:
        now = get_current_time_et()
        return now.hour < 15 or (now.hour == 15 and now.minute < 30)
    except Exception as e:
        logger.error(f"[Day Trade Time Check Error] {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def should_exit_day_trades():
    try:
        now = get_current_time_et()
        return now.hour == 15 and now.minute >= 55
    except Exception as e:
        logger.error(f"[Day Trade Exit Time Error] {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def place_order(symbol, qty, side, type="market", time_in_force="gtc"):
    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        logger.info(f"🟢 Order Placed: {side.upper()} {qty} {symbol}")
        send_telegram_message(f"🟢 Order Placed: `{side.upper()} {qty} {symbol}`")
        return {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'entry_time': datetime.utcnow().isoformat(),
            'order_id': order.id
        }
    except Exception as e:
        logger.error(f"[Order Placement Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"❌ Order Failed: `{side.upper()} {qty} {symbol}`\nReason: `{str(e)}`")
        return None


def retry_order_placement(symbol, qty, side, type="market", time_in_force="gtc",
                          retries=MAX_ORDER_RETRIES, delay=RETRY_DELAY_SECONDS):
    attempt = 1
    while attempt <= retries:
        logger.info(f"📦 Attempt {attempt} placing order: {side.upper()} {qty} {symbol}")
        result = place_order(symbol, qty, side, type, time_in_force)
        if result is not None:
            logger.info(f"✅ Order succeeded on attempt {attempt}: {symbol}")
            return result
        else:
            logger.warning(f"🔁 Order failed on attempt {attempt} for {symbol}. Retrying in {delay}s...")
            time.sleep(delay)
            attempt += 1

    logger.error(f"❌ All {retries} attempts failed for {symbol}. Giving up.")
    send_telegram_message(f"❌ All {retries} order attempts failed for `{symbol}`.")
    return None


def close_position(symbol):
    try:
        alpaca.close_position(symbol)
        logger.info(f"🔴 Closed position: {symbol}")
        send_telegram_message(f"🔴 Closed position: `{symbol}`")
    except Exception as e:
        logger.error(f"[Close Position Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"⚠️ Failed to close `{symbol}`\nReason: `{str(e)}`")


def manage_open_positions(positions):
    try:
        for pos in positions:
            symbol = pos['symbol']
            try:
                if is_day_trade(pos) and should_exit_day_trades():
                    logger.info(f"⏰ Time to close day trade: {symbol}")
                    close_position(symbol)
            except Exception as e:
                logger.error(f"[Error Managing {symbol}] {str(e)}")
                logger.debug(traceback.format_exc())
                send_telegram_message(f"⚠️ Error managing `{symbol}`\nReason: `{str(e)}`")
    except Exception as e:
        logger.critical(f"[manage_open_positions Failed] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message("🚨 Failed to manage open positions.")


def get_open_positions():
    try:
        return [p._raw for p in alpaca.list_positions() if p.qty != '0']
    except Exception as e:
        logger.error(f"[Get Open Positions Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message("⚠️ Could not fetch open positions.")
        return []