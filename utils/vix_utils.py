# vix_utils.py

import requests
from datetime import datetime, timedelta
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ENABLE_VIX_THROTTLING,
    VIX_MAX_THRESHOLD,
    VIX_MODERATE_THRESHOLD,
)
from utils.logger import bot_logger

def fetch_vix_price():
    """
    Fetches the most recent VIX index value using Alpaca's market data API.
    Returns float VIX value or None if unavailable.
    """
    try:
        endpoint = f"{ALPACA_BASE_URL}/v2/stocks/VIX/quotes/latest"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }

        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        vix_price = float(data["ask_price"])  # fallback to 'ask' for latest
        bot_logger.debug(f"📈 Current VIX: {vix_price}")
        return vix_price

    except Exception as e:
        bot_logger.warning(f"[VIX Fetch Error] {str(e)}")
        return None


def should_throttle_trades(vix_value: float) -> bool:
    """
    Returns True if trading should be throttled (i.e., VIX too high).
    """
    if not ENABLE_VIX_THROTTLING or vix_value is None:
        return False

    if vix_value >= VIX_MAX_THRESHOLD:
        bot_logger.warning(f"🚫 VIX ({vix_value}) exceeds threshold ({VIX_MAX_THRESHOLD}) — Throttling trades")
        return True

    return False


def is_vix_moderately_high(vix_value: float) -> bool:
    """
    Returns True if VIX is above moderate threshold.
    """
    if vix_value is None:
        return False

    return vix_value >= VIX_MODERATE_THRESHOLD