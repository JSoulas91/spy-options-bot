def calculate_confidence_score(signals):
    """
    Accepts a dictionary of signals like:
    {
        'trend_confirmed': True,
        'volume_spike': True,
        'above_vwap': True,
        'news_sentiment': 0.8,
        'ml_prediction': 1,
        'regime': 'bullish_trend'
    }
    Returns a confidence score between 0 and 100
    """
    score = 0
    weights = {
        'trend_confirmed': 20,
        'volume_spike': 20,
        'above_vwap': 15,
        'news_sentiment': 15,
        'ml_prediction': 20,
        'regime': 10
    }

    if signals.get('trend_confirmed'):
        score += weights['trend_confirmed']
    if signals.get('volume_spike'):
        score += weights['volume_spike']
    if signals.get('above_vwap'):
        score += weights['above_vwap']
    if isinstance(signals.get('news_sentiment'), (int, float)):
        score += int(signals['news_sentiment'] * weights['news_sentiment'])
    if signals.get('ml_prediction') == 1:
        score += weights['ml_prediction']
    if signals.get('regime') in ['bullish_trend', 'bearish_trend']:
        score += weights['regime']
    
    return min(score, 100)
