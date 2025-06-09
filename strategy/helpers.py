# strategy/helpers.py

def is_day_trade(position: dict) -> bool:
    """
    Determines if the position is a day trade based on its type or metadata.
    Falls back to detecting if 'day' appears in the trade_type field.
    """
    trade_type = position.get("trade_type", "").lower()
    return "day" in trade_type or trade_type == "scalp"

def is_swing_trade(position: dict) -> bool:
    """
    Determines if the position is a swing trade.
    Returns True if trade_type is explicitly marked as 'swing'.
    """
    trade_type = position.get("trade_type", "").lower()
    return "swing" in trade_type