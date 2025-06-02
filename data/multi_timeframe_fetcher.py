# data/multi_timeframe_fetcher.py
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def fetch_timeframes(symbol="SPY"):
    now = datetime.utcnow()

    # Define all timeframes and lookback windows
    timeframes = {
        "5m_5d": (TimeFrame(5), now - timedelta(days=5)),
        "15m_10d": (TimeFrame(15), now - timedelta(days=10)),
        "1h_15d": (TimeFrame.Hour, now - timedelta(days=15)),
        "1d_1m": (TimeFrame.Day, now - timedelta(days=30)),
        "1d_3m": (TimeFrame.Day, now - timedelta(days=90)),
        "1d_6m": (TimeFrame.Day, now - timedelta(days=180)),
    }

    result = {}
    for label, (tf, start) in timeframes.items():
        try:
            bars = api.get_bars(symbol, tf, start=start, end=now).df
            result[label] = bars
        except Exception as e:
            print(f"Error fetching {label}: {e}")
            result[label] = None

    return result