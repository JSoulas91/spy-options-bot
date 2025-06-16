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

def calculate_atr(df: pd.DataFrame, period: int = 14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df: pd.DataFrame, period: int = 14):
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_indicators(df: pd.DataFrame):
    df = df.copy()

    # Ensure required columns
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    # Compute VWAP internally
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    price = df["vwap"]

    df["ema_20"] = calculate_ema(price, 20)
    df["rsi_14"] = calculate_rsi(price, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(price)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(price, 20)
    df["atr_14"] = calculate_atr(df, 14)
    df["adx_14"] = calculate_adx(df, 14)

    return df

def compute_trade_indicators(vwap: float, volume: float) -> dict:
    """
    Create a synthetic 30-bar DataFrame with estimated OHLC values from VWAP.
    """
    num_bars = 30
    close_series = np.append(np.full(num_bars - 1, np.nan), vwap)
    volume_series = np.append(np.full(num_bars - 1, np.nan), volume)

    # Assume close = high = low = open = vwap (best-effort for synthetic indicators)
    df = pd.DataFrame({
        'open': close_series,
        'high': close_series,
        'low': close_series,
        'close': close_series,
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