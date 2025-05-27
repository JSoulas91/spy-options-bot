from helpers import is_day_trade, is_swing_trade

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
        if market_data['price'] >= position['entry_price'] * 1.05:
            print("Day trade hit profit target. Exiting.")
            action = "exit"
        elif market_data['price'] <= position['entry_price'] * 0.95:
            print("Day trade hit stop-loss. Exiting.")
            action = "exit"

    elif is_swing_trade(position):
        # Example logic for swing trade
        if market_data['price'] >= position['entry_price'] * 1.15:
            print("Swing trade profit target hit.")
            action = "exit"
        elif market_data['price'] <= position['entry_price'] * 0.90:
            print("Swing trade stop-loss hit.")
            action = "exit"

    return action