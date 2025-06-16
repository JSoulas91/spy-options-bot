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

def compute_trade_indicators(open_price, high, low, close, volume) -> dict:
    """
    Create a fake 30-bar DataFrame with the current OHLCV values as the latest bar,
    allowing talib-based indicators to be computed.
    """
    num_bars = 30
    df = pd.DataFrame({
        'open': np.append(np.full(num_bars - 1, np.nan), open_price),
        'high': np.append(np.full(num_bars - 1, np.nan), high),
        'low': np.append(np.full(num_bars - 1, np.nan), low),
        'close': np.append(np.full(num_bars - 1, np.nan), close),
        'volume': np.append(np.full(num_bars - 1, np.nan), volume),
    })

    try:
        df = calculate_indicators(df)
        latest = df.iloc[-1]

        ema = latest['EMA_20'] if not pd.isna(latest['EMA_20']) and latest['EMA_20'] != 0 else close
        sma_ratio = close / ema
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        volume_zscore = (volume - volume_mean) / volume_std if volume_std > 0 else 0

        return {
            "rsi": float(latest['RSI_14']) if not pd.isna(latest['RSI_14']) else 50.0,
            "macd": float(latest['MACD']) if not pd.isna(latest['MACD']) else 0.0,
            "sma_ratio": float(sma_ratio),
            "volume_zscore": float(volume_zscore)
        }
    except Exception:
        return {
            "rsi": 50.0,
            "macd": 0.0,
            "sma_ratio": 1.0,
            "volume_zscore": 0.0
        }