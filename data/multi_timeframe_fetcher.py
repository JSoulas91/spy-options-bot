import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.logger import bot_logger

TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return {}

    df = df.copy()

    # EMA 20, 50, 200
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI (14)
    df["rsi"] = compute_rsi(df["close"], period=14)

    # MACD & Signal
    df["macd"], df["macd_signal"] = compute_macd(df["close"])

    # ATR (14)
    df["atr"] = compute_atr(df, period=14)

    # VWAP (Volume Weighted Avg Price)
    df["vwap"] = compute_vwap(df)

    # Bollinger Bands (20, 2 std)
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    # Support and Resistance (simple pivot points)
    support, resistance = compute_support_resistance(df)
    
    last_row = df.iloc[-1]
    return {
        "price": last_row["close"],
        "ema_20": last_row["ema_20"],
        "ema_50": last_row["ema_50"],
        "ema_200": last_row["ema_200"],
        "rsi": last_row["rsi"],
        "macd": last_row["macd"],
        "macd_signal": last_row["macd_signal"],
        "atr": last_row["atr"],
        "vwap": last_row["vwap"],
        "bb_upper": last_row["bb_upper"],
        "bb_lower": last_row["bb_lower"],
        "support": support,
        "resistance": resistance
    }

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta).clip(lower=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_vwap(df):
    cum_vol = df["volume"].cumsum()
    cum_vol_price = (df["close"] * df["volume"]).cumsum()
    vwap = cum_vol_price / (cum_vol + 1e-9)
    return vwap

def compute_support_resistance(df):
    # Simple method: use pivot points from last 14 days
    pivots = df['close'].rolling(window=14)
    support = pivots.min().iloc[-1] if not pivots.min().empty else None
    resistance = pivots.max().iloc[-1] if not pivots.max().empty else None
    return support, resistance

def fetch_tradier_timesales(symbol, start_dt, end_dt, interval):
    url = f"{TRADIER_BASE_URL}/markets/timesales"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M"),
        "session_filter": "open"
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            bot_logger.warning(f"[Tradier Timesales {interval}] Failed: {response.status_code}")
            return None
        data = response.json()
        if "series" not in data or "data" not in data["series"]:
            bot_logger.warning(f"[Tradier Timesales {interval}] No data")
            return None
        df = pd.DataFrame(data["series"]["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        bot_logger.exception(f"[Tradier Timesales {interval}] {e}")
        return None

def fetch_tradier_history(symbol, start_date, end_date):
    url = f"{TRADIER_BASE_URL}/markets/history"
    params = {
        "symbol": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "interval": "daily"
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            bot_logger.warning(f"[Tradier History] Failed for {symbol}: {response.status_code}")
            return None
        data = response.json()
        if "history" not in data or data["history"] is None:
            bot_logger.warning(f"[Tradier History] No data for {symbol}")
            return None
        quotes = data["history"].get("day", [])
        if not quotes:
            return None
        df = pd.DataFrame(quotes)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        bot_logger.exception(f"[Tradier History] {e}")
        return None

def get_multi_timeframe_data(symbol="SPY"):
    now = datetime.utcnow()

    # Fetch longer-term daily OHLC with indicators for different lookbacks
    lookbacks = {
        "5d": now.date() - timedelta(days=5),
        "10d": now.date() - timedelta(days=10),
        "15d": now.date() - timedelta(days=15),
        "1mo": now.date() - timedelta(days=30),
        "3mo": now.date() - timedelta(days=90),
        "6mo": now.date() - timedelta(days=180),
    }

    daily_features = {}
    for label, start_date in lookbacks.items():
        try:
            df = fetch_tradier_history(symbol, start_date, now.date())
            daily_features[label] = compute_indicators(df) if df is not None else {}
        except Exception as e:
            bot_logger.exception(f"[Tradier History Error] {label}: {e}")
            daily_features[label] = {}

    # Fetch intraday OHLC with indicators
    intraday_intervals = {
        "1min_5d":  {"interval": "1min",  "start": now - timedelta(days=5)},
        "5min_5d":  {"interval": "5min",  "start": now - timedelta(days=5)},
        "15min_15d": {"interval": "15min", "start": now - timedelta(days=15)},
        "1hr_30d":  {"interval": "1hour", "start": now - timedelta(days=30)},
        "1d_6mo":   {"interval": "daily", "start": now - timedelta(days=180)},
    }

    intraday_features = {}
    for label, config in intraday_intervals.items():
        try:
            df = fetch_tradier_timesales(symbol, config["start"], now, config["interval"])
            intraday_features[label] = compute_indicators(df) if df is not None else {}
        except Exception as e:
            bot_logger.exception(f"[Tradier Intraday Error] {label}: {e}")
           