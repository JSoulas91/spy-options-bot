# data/quote_utils.py

import time
import requests
from config import TRADIER_BASE_URL, TRADIER_ACCESS_TOKEN
from utils.logger import bot_logger as logger
from utils.cache_manager import cache

HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json",
}


def get_spy_quote(ttl: int = 5) -> float:
    """
    Returns the latest SPY quote, using in-memory cache to limit API calls.
    Default TTL is 5 seconds.
    """
    cache_key = "spy_quote"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        response = requests.get(
            f"{TRADIER_BASE_URL}/v1/markets/quotes",
            params={"symbols": "SPY"},
            headers=HEADERS,
            timeout=3,
        )
        data = response.json()

        quote = data.get("quotes", {}).get("quote", {})
        price = quote.get("last")

        if price:
            cache.set(cache_key, price, ttl=ttl)
            return price
        else:
            raise ValueError(f"Invalid quote data: {quote}")

    except Exception as e:
        logger.warning(f"[QuoteUtils] Failed to fetch SPY quote: {e}")
        return -1  # fallback: caller should handle this safely