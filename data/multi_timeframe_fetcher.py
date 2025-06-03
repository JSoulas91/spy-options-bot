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

def fetch_tradier_history(symbol, start_date, end_date):
    """
    Fetch historical daily OHLC data from Tradier.
    """
    url = f"{TRADIER_BASE_URL}/markets/history"
    params = {
        "symbol": symbol,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "interval": "daily"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        bot_logger.warning(f"[Tradier Fetch] Failed for {symbol}: {response.status_code}")
        return None

    data = response.json()
    if "history" not in data or data["history"] is None:
        bot_logger.warning(f"[Tradier Fetch] No history returned for {symbol}")
        return None

    quotes = data["history"].get("day", [])
    if not quotes:
        return None

    df = pd.DataFrame(quotes)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    return df

def fetch_long_term_features(symbol="SPY"):
    now = datetime.utcnow().date()

    timeframes = {
        "5d": now - timedelta(days=5),
        "10d": now - timedelta(days=10),
        "15d": now - timedelta(days=15),
        "1mo": now - timedelta(days=30),
        "3mo": now - timedelta(days=90),
        "6mo": now - timedelta(days=180),
    }

    results = {}
    for label, start_date in timeframes.items():
        try:
            df = fetch_tradier_history(symbol, start_date, now)
            if df is not None and not df.empty:
                features = compute_indicators(df)
                results[label] = features if features else {}
            else:
                results[label] = {}
        except Exception as e:
            bot_logger.exception(f"[Tradier Fetch Error] {label}: {e}")
            results[label] = {}

    return results