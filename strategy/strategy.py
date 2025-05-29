import traceback
from helpers import is_day_trade, is_swing_trade
from config import (
    CONFIDENCE_THRESHOLD,
    STOP_LOSS_ATR_MULTIPLIER,
    TRAILING_STOP_PERCENT,
    PREFERS_LIQUID_OPTIONS,
    USE_AGGRESSIVE_MODE,
    AGGRESSIVE_TRADE_SIZE,
    DEFAULT_POSITION_SIZE
)
from utils.logger import bot_logger as logger

def evaluate_trade(position, market_data):
    """
    Apply strategy logic to current open position based on trade type.
    :param position: dict containing current open position info
    :param market_data: dict of latest price, indicators, etc.
    :return: 'hold', 'exit', or 'scale'
    """
    try:
        action = "hold"
        entry_price = position.get('entry_price')
        price = market_data.get('price')

        if entry_price is None or price is None:
            raise ValueError("Missing 'entry_price' or 'price' in input data.")

        logger.debug(f"[Strategy] Evaluating trade — Entry: {entry_price}, Current: {price}")

        if is_day_trade(position):
            logger.debug("[Strategy] Trade type: Day Trade")
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                logger.info("✅ Day trade hit profit target. Exiting.")
                action = "exit"
            elif price <= entry_price * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.05):
                logger.info("🛑 Day trade hit stop-loss. Exiting.")
                action = "exit"

        elif is_swing_trade(position):
            logger.debug("[Strategy] Trade type: Swing Trade")
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT * 1.5):
                logger.info("✅ Swing trade hit profit target. Exiting.")
                action = "exit"
            elif price <= entry_price * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.06):
                logger.info("🛑 Swing trade hit stop-loss. Exiting.")
                action = "exit"

        logger.debug(f"[Strategy] Action determined: {action}")
        return action

    except Exception as e:
        logger.error(f"[Strategy Error] {str(e)}")
        logger.debug(traceback.format_exc())
        return "hold"  # Safe fallback