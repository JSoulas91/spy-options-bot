# meta/meta_state.py

def build_meta_state_for_entry(market_data, confidence_score, trade_type=0):
    """
    Constructs the meta-state for PPO entry decision-making.
    Meta-state: [confidence_score, vix, hour, trade_type, volatility]
    trade_type: 0 = day trade, 1 = swing trade
    """
    try:
        vix = market_data.get("vix", 0)
        timestamp = market_data.get("timestamp", "")
        hour = int(timestamp.split(" ")[1].split(":")[0]) if timestamp else 12
        volatility = market_data.get("indicators", {}).get("atr", 0)
        return [confidence_score, vix or 0, hour, trade_type, volatility]
    except Exception:
        return [confidence_score, 0, 12, trade_type, 0]

def build_meta_state_for_exit(market_data, confidence_score, trade_type=0, trade_duration_minutes=0, current_profit=0):
    """
    Constructs the meta-state for PPO exit decision-making.
    Meta-state: [confidence_score, vix, trade_duration_minutes, trade_type, current_profit, volatility]
    """
    try:
        vix = market_data.get("vix", 0)
        volatility = market_data.get("indicators", {}).get("atr", 0)
        return [
            confidence_score,
            vix or 0,
            trade_duration_minutes,
            trade_type,
            current_profit,
            volatility
        ]
    except Exception:
        return [
            confidence_score,
            0,
            trade_duration_minutes,
            trade_type,
            current_profit,
            0
        ]