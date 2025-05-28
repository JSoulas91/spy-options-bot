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

def evaluate_trade(position, market_data):
    """
    Apply strategy logic to current open position based on trade type.
    :param position: dict containing current open position info
    :param market_data: dict of latest price, indicators, etc.
    :return: 'hold', 'exit', or 'scale'
    """
    action = "hold"

    if is_day_trade(position):
        # Example logic for day trade
        if market_data['price'] >= position['entry_price'] * (1 + TRAILING_STOP_PERCENT):
            print("Day trade hit profit target. Exiting.")
            action = "exit"
        elif market_data['price'] <= position['entry_price'] * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.05):
            print("Day trade hit stop-loss. Exiting.")
            action = "exit"

    elif is_swing_trade(position):
        # Example logic for swing trade
        if market_data['price'] >= position['entry_price'] * (1 + TRAILING_STOP_PERCENT * 1.5):
            print("Swing trade profit target hit.")
            action = "exit"
        elif market_data['price'] <= position['entry_price'] * (1 - STOP_LOSS_ATR_MULTIPLIER * 0.06):
            print("Swing trade stop-loss hit.")
            action = "exit"

    return action