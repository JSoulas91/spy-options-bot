import pandas as pd
import numpy as np

def calculate_ema(data: pd.Series, period: int):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series):
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data: pd.Series, period: int = 20):
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    return upper, middle, lower

def calculate_atr(data: pd.Series, period: int = 14):
    vwap_high = data.rolling(window=2).max()
    vwap_low = data.rolling(window=2).min()
    tr = vwap_high - vwap_low
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(data: pd.Series, period: int = 14):
    up_move = data.diff().clip(lower=0)
    down_move = -data.diff().clip(upper=0)
    tr = data.diff().abs()
    plus_dm = up_move.rolling(period).mean()
    minus_dm = down_move.rolling(period).mean()
    tr14 = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm / (tr14 + 1e-9))
    minus_di = 100 * (minus_dm / (tr14 + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()
    return adx

def calculate_indicators(data: pd.DataFrame):
    df = data.copy()
    price = df["vwap"]

    df["ema_20"] = calculate_ema(price, 20)
    df["rsi_14"] = calculate_rsi(price, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(price)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(price, 20)
    df["atr_14"] = calculate_atr(price, 14)
    df["adx_14"] = calculate_adx(price, 14)

    return df

def compute_trade_indicators(vwap: float, volume: float) -> dict:
    """
    Create a synthetic 30-bar DataFrame with the latest vwap + volume,
    allowing indicators to be calculated on sparse history.
    """
    num_bars = 30
    vwap_series = np.append(np.full(num_bars - 1, np.nan), vwap)
    volume_series = np.append(np.full(num_bars - 1, np.nan), volume)

    df = pd.DataFrame({
        'vwap': vwap_series,
        'volume': volume_series,
    })

    try:
        df = calculate_indicators(df)
        latest = df.iloc[-1]

        ema = latest['ema_20'] if not pd.isna(latest['ema_20']) and latest['ema_20'] != 0 else vwap
        sma_ratio = vwap / ema
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        volume_zscore = (volume - volume_mean) / volume_std if volume_std > 0 else 0

        return {
            "rsi": float(latest['rsi_14']) if not pd.isna(latest['rsi_14']) else 50.0,
            "macd": float(latest['macd']) if not pd.isna(latest['macd']) else 0.0,
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