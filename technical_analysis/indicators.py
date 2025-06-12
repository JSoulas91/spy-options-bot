import pandas as pd
import numpy as np
import talib

def calculate_ema(data, period):
    return talib.EMA(data['close'], timeperiod=period)

def calculate_rsi(data, period=14):
    return talib.RSI(data['close'], timeperiod=period)

def calculate_macd(data):
    macd, signal, hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, signal, hist

def calculate_bollinger_bands(data, period=20):
    upper, middle, lower = talib.BBANDS(data['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    return upper, middle, lower

def calculate_vwap(data):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    cumulative_tp_volume = (typical_price * data['volume']).cumsum()
    cumulative_volume = data['volume'].cumsum()
    vwap = cumulative_tp_volume / cumulative_volume
    return vwap

def calculate_atr(data, period=14):
    return talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)

def calculate_adx(data, period=14):
    return talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)

def calculate_indicators(data):
    df = data.copy()
    df['EMA_20'] = calculate_ema(df, 20)
    df['RSI_14'] = calculate_rsi(df, 14)
    macd, signal, hist = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    upper, middle, lower = calculate_bollinger_bands(df, 20)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    df['VWAP'] = calculate_vwap(df)
    df['ATR_14'] = calculate_atr(df, 14)
    df['ADX_14'] = calculate_adx(df, 14)
    return df