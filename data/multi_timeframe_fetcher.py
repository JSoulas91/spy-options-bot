# data/multi_timeframe_fetcher.py

from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
import pandas as pd
import numpy as np

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["macd"], df["macd_signal"] = compute_macd(df["close"])

    last_row = df.iloc[-1]
    return {
        "price": last_row["close"],
        "ema_20": last_row["ema_20"],
        "rsi": last_row["rsi"],
        "macd": last_row["macd"]
    }

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_long_term_features(symbol="SPY"):
    now = datetime.utcnow()

    timeframes = {
        "5d": (TimeFrame(5), now - timedelta(days=5)),
        "10d": (TimeFrame(15), now - timedelta(days=10)),
        "15d": (TimeFrame.Hour, now - timedelta(days=15)),
        "1mo": (TimeFrame.Day, now - timedelta(days=30)),
        "3mo": (TimeFrame.Day, now - timedelta(days=90)),
        "6mo": (TimeFrame.Day, now - timedelta(days=180)),
    }

    results = {}
    for label, (tf, start) in timeframes.items():
        try:
            bars = api.get_bars(symbol, tf, start=start, end=now).df
            if not bars.empty:
                bars = bars[bars.index >= start]
                features = compute_indicators(bars)
                results[label] = features if features else {}
            else:
                results[label] = {}
        except Exception as e:
            print(f"Error fetching {label}: {e}")
            results[label] = {}

    return results