# strategy/helpers.py

def is_day_trade(position: dict) -> bool:
    """Detect if position is a day trade."""
    return position.get("trade_type", "").lower() == "day"

def is_swing_trade(position: dict) -> bool:
    """Detect if position is a swing trade."""
    return position.get("trade_type", "").lower() == "swing"

def get_min_meta_confidence(regime: str) -> float:
    """Return regime-adjusted minimum confidence for meta-agent exit."""
    if regime == "bull":
        return 0.25
    if regime == "bear":
        return 0.35
    return 0.30  # vol_cluster default