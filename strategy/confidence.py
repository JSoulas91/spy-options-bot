def calculate_confidence_score(signal_data):
    """
    Calculates a confidence score (0–100) based on multiple trade factors.
    Expects signal_data to be a dictionary with the following boolean or numeric keys:
        - 'trend_alignment' (bool)
        - 'vwap_confirmation' (bool)
        - 'support_resistance_nearby' (bool)
        - 'volume_spike' (bool)
        - 'news_sentiment_score' (float, -1 to 1)
        - 'regime' (str: bullish, bearish, sideways, choppy)
    """
    score = 0

    if signal_data.get('trend_alignment'):
        score += 25
    if signal_data.get('vwap_confirmation'):
        score += 20
    if signal_data.get('support_resistance_nearby'):
        score += 15
    if signal_data.get('volume_spike'):
        score += 15

    sentiment_score = signal_data.get('news_sentiment_score', 0)
    score += max(0, min(15, sentiment_score * 15))  # scale -1 to 1 → -15 to +15

    regime = signal_data.get('regime', '')
    if regime == 'bullish':
        score += 10
    elif regime == 'bearish':
        score += 5
    elif regime == 'sideways':
        score -= 5
    elif regime == 'choppy':
        score -= 10

    return max(0, min(100, score))
