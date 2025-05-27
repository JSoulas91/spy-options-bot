import numpy as np
import talib

def detect_market_regime(data):
    macd, signal, hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    
    # Strong trend if ADX > 25 and MACD histogram diverging
    if adx.iloc[-1] > 25:
        if hist.iloc[-1] > 0:
            return 'bullish_trend'
        elif hist.iloc[-1] < 0:
            return 'bearish_trend'
    return 'sideways'

def is_safe_to_trade(regime):
    return regime in ['bullish_trend', 'bearish_trend']
