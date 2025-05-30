# confidence.py

def calculate_confidence_score(signal_data):
    """
    Calculates a confidence score (0–100) based on multiple trade factors.
    
    Input:
        signal_data (dict): Expected keys:
            - trend_alignment (bool)
            - vwap_confirmation (bool)
            - near_support (bool)
            - near_resistance (bool)
            - volume_spike (bool)
            - momentum_up (bool) — optional
            - exit_pressure (bool) — optional
            - news_sentiment_score (float) — range -1.0 to +1.0
            - regime (str) — one of: bullish, bearish, sideways, choppy
    
    Returns:
        int: Confidence score between 0 and 100
    """
    score = 0

    # Core signal alignment
    if signal_data.get("trend_alignment", False):
        score += 25
    if signal_data.get("vwap_confirmation", False):
        score += 20
    if signal_data.get("near_support", False):
        score += 10
    if signal_data.get("near_resistance", False):
        score -= 10
    if signal_data.get("volume_spike", False):
        score += 15

    # Optional extras
    if signal_data.get("momentum_up", False):
        score += 10
    if signal_data.get("exit_pressure", False):
        score -= 15

    # News sentiment
    sentiment_score = signal_data.get("news_sentiment_score", 0.0)
    score += round(max(-15, min(15, sentiment_score * 15)))  # normalize to ±15

    # Market regime weight
    regime = signal_data.get("regime", "").lower()
    if regime == "bullish":
        score += 10
    elif regime == "bearish":
        score += 5
    elif regime == "sideways":
        score -= 5
    elif regime == "choppy":
        score -= 10

    return max(0, min(100, score))