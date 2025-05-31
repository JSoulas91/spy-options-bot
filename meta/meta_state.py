# meta/meta_state.py

def build_meta_state_for_entry(market_data, confidence_score):
    """
    Constructs the meta-state for PPO entry decision-making.
    Meta-state: [confidence_score, vix, hour, trade_type (0=day), volatility]
    """
    try:
        vix = market_data.get("vix", 0)
        timestamp = market_data.get("timestamp", "")
        hour = int(timestamp.split(" ")[1].split(":")[0]) if timestamp else 12
        volatility = market_data.get("indicators", {}).get("atr", 0)
        return [confidence_score, vix or 0, hour, 0, volatility]
    except Exception:
        return [confidence_score, 0, 12, 0, 0]