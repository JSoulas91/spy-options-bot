import pandas as pd
import numpy as np

def detect_market_regime(df):
    """
    Detects current market regime using moving average slope and volatility.
    Returns one of: 'bullish', 'bearish', or 'sideways'
    """
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['volatility'] = df['close'].rolling(window=14).std()

    recent_slope = df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]
    price_above_200 = df['close'].iloc[-1] > df['sma_200'].iloc[-1]
    volatility = df['volatility'].iloc[-1]

    # Rules
    if recent_slope > 0 and price_above_200:
        return 'bullish'
    elif recent_slope < 0 and not price_above_200:
        return 'bearish'
    else:
        if volatility < 2:
            return 'sideways'
        else:
            return 'choppy'
