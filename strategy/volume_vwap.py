import pandas as pd

def calculate_vwap(df):
    """
    Adds VWAP column to the given DataFrame.
    VWAP = sum(price * volume) / sum(volume)
    """
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vol_price'] = (df['typical_price'] * df['volume']).cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_volume']
    return df

def vwap_confirmation(df):
    """
    Confirm bullish or bearish trend relative to VWAP.
    """
    df = calculate_vwap(df)
    latest_price = df.iloc[-1]['close']
    latest_vwap = df.iloc[-1]['vwap']

    if latest_price > latest_vwap:
        return "bullish"
    elif latest_price < latest_vwap:
        return "bearish"
    else:
        return "neutral"

def is_vwap_confirmed(df, direction="bullish"):
    """
    Boolean version of VWAP confirmation used in confidence scoring.
    """
    df = calculate_vwap(df)
    price = df.iloc[-1]['close']
    vwap = df.iloc[-1]['vwap']
    if direction == "bullish":
        return price > vwap
    elif direction == "bearish":
        return price < vwap
    return False

def price_relative_to_vwap(df):
    """
    Percentage distance from VWAP (can be used for overbought/oversold logic).
    """
    df = calculate_vwap(df)
    price = df.iloc[-1]['close']
    vwap = df.iloc[-1]['vwap']
    return round((price - vwap) / vwap * 100, 2)

def vwap_slope(df, periods=3):
    """
    Returns the slope of VWAP over the last N candles.
    Positive = trending up, Negative = trending down.
    """
    df = calculate_vwap(df)
    recent_vwap = df['vwap'].iloc[-periods:]
    return recent_vwap.diff().mean()