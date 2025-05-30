import pandas as pd
import numpy as np

def detect_market_regime(df, slope_threshold=0, low_vol_threshold=2.0):
    """
    Detects current market regime using moving average slope and volatility.
    Returns one of: 'bullish', 'bearish', 'sideways', or 'choppy'
    """
    if len(df) < 210:
        return 'unknown'  # not enough data for reliable regime detection

    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['volatility'] = df['close'].rolling(window=14).std()

    recent_slope = df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]
    price_above_200 = df['close'].iloc[-1] > df['sma_200'].iloc[-1]
    volatility = df['volatility'].iloc[-1]

    # Regime detection logic
    if recent_slope > slope_threshold and price_above_200 and volatility <= low_vol_threshold * 1.5:
        return 'bullish'
    elif recent_slope < -slope_threshold and not price_above_200 and volatility <= low_vol_threshold * 1.5:
        return 'bearish'
    elif volatility < low_vol_threshold:
        return 'sideways'
    else:
        return 'choppy'