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


def price_relative_to_vwap(df):
    df = calculate_vwap(df)
    price = df.iloc[-1]['close']
    vwap = df.iloc[-1]['vwap']
    return round((price - vwap) / vwap * 100, 2)  # Percentage above/below
