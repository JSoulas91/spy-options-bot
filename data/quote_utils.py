import time
from typing import Optional
from data.tradier_api import get_equity_quote as fetch_spy_quote

# ─────────────────────────────────────────────
# ░▒▓  In-memory caching for quotes  ▓▒░
# ─────────────────────────────────────────────
QUOTE_CACHE_SECONDS = 6
_last_quote_time: float = 0.0
_last_quote_data: Optional[float] = None

def get_spy_quote() -> Optional[float]:
    """
    Returns the latest SPY quote (last price) with in-memory caching for QUOTE_CACHE_SECONDS.
    Reduces redundant Tradier API calls while preserving quote freshness.
    """
    global _last_quote_time, _last_quote_data
    now = time.time()

    # Return cached quote if within cache window
    if now - _last_quote_time < QUOTE_CACHE_SECONDS and _last_quote_data is not None:
        return _last_quote_data

    # Fetch fresh quote from Tradier
    quote = fetch_spy_quote()
    if isinstance(quote, dict):
        last_price = quote.get("last")
        if last_price:
            _last_quote_data = last_price
            _last_quote_time = now
            return last_price

    # On failure, return last cached quote if available
    return _last_quote_data