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
from utils.logger import bot_logger  # ✅ Import logger

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

        if not entry_price or not price:
            raise ValueError("Missing 'entry_price' or 'price' in input data.")

        if is_day_trade(position):
            # Day trade logic
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                bot_logger.info("Day trade hit profit target. Exiting.")
                action = "exit"
            elif price <= entry_price * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.05):
                bot_logger.info("Day trade hit stop-loss. Exiting.")
                action = "exit"

        elif is_swing_trade(position):
            # Swing trade logic
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT * 1.5):
                bot_logger.info("Swing trade profit target hit. Exiting.")
                action = "exit"
            elif price <= entry_price * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.06):
                bot_logger.info("Swing trade stop-loss hit. Exiting.")
                action = "exit"

        return action

    except Exception as e:
        error_message = f"[Strategy Error] {str(e)}"
        bot_logger.error(error_message)
        bot_logger.debug(traceback.format_exc())
        return "hold"  # Safe fallback