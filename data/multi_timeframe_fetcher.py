# data/multi_timeframe_fetcher.py
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def fetch_timeframes(symbol="SPY"):
    now = datetime.utcnow()
    start = now - timedelta(days=2)

    return {
        "5m": api.get_bars(symbol, TimeFrame(5), start=start, end=now).df,
        "15m": api.get_bars(symbol, TimeFrame(15), start=start, end=now).df,
        "1h": api.get_bars(symbol, TimeFrame.Hour, start=start, end=now).df
    }