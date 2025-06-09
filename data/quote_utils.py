# data/quote_utils.py

import time
from typing import Optional
from data.tradier_api import get_equity_quote as fetch_spy_quote

# Cache settings
QUOTE_CACHE_SECONDS = 6
_last_quote_time = 0.0
_last_quote_data: Optional[float] = None

def get_spy_quote() -> Optional[float]:
    """
    Returns the latest SPY quote with in-memory caching (valid for QUOTE_CACHE_SECONDS).
    Reduces Tradier API usage while maintaining accuracy.
    """
    global _last_quote_time, _last_quote_data
    now = time.time()

    if now - _last_quote_time < QUOTE_CACHE_SECONDS and _last_quote_data is not None:
        return _last_quote_data

    quote = fetch_spy_quote()
    if quote:
        _last_quote_data = quote
        _last_quote_time = now
        return quote

    return _last_quote_data  # Fallback to last known