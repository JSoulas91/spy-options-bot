"""
Lightweight VIX helper that no longer depends on Alpaca.
Fetches the latest VIX value from Yahoo Finance’s free quote API.
"""

import os
import requests
from dotenv import load_dotenv
from utils.logger import bot_logger
from config import (
    ENABLE_VIX_THROTTLING,
    VIX_MAX_THRESHOLD,
    VIX_MODERATE_THRESHOLD,
)

# Load .env variables (e.g., SIMULATION_MODE)
load_dotenv()

# Encoded “^VIX” for the Yahoo Finance quote endpoint
VIX_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=%5EVIX"

# Default fallback VIX value for simulation mode
DEFAULT_VIX = 20.0


def fetch_vix_price() -> float | None:
    """
    Fetches the most recent VIX index value using Yahoo Finance.

    Returns
    -------
    float | None
        Latest VIX price, or None if unavailable.
    """
    if os.getenv("SIMULATION_MODE", "").lower() == "true":
        bot_logger.debug("🔁 SIMULATION_MODE enabled – using default VIX = %.2f", DEFAULT_VIX)
        return DEFAULT_VIX

    try:
        resp = requests.get(VIX_QUOTE_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("quoteResponse", {}).get("result", [])
        if not result:
            raise ValueError("No quote data returned")

        vix_price = float(result[0]["regularMarketPrice"])
        bot_logger.debug("📈 Current VIX (Yahoo): %.2f", vix_price)
        return vix_price

    except Exception as exc:
        bot_logger.warning("[VIX Fetch Error] %s", exc)
        return None


def should_throttle_trades(vix_value: float | None) -> bool:
    """
    Returns True if trading should be throttled (VIX too high).

    Parameters
    ----------
    vix_value : float | None
        Latest VIX value.

    Returns
    -------
    bool
    """
    if not ENABLE_VIX_THROTTLING or vix_value is None:
        return False

    if vix_value >= VIX_MAX_THRESHOLD:
        bot_logger.warning(
            "🚫 VIX (%.2f) exceeds max threshold (%.2f) — throttling trades",
            vix_value,
            VIX_MAX_THRESHOLD,
        )
        return True

    return False


def is_vix_moderately_high(vix_value: float | None) -> bool:
    """
    Returns True if VIX is above the moderate threshold.

    Parameters
    ----------
    vix_value : float | None
        Latest VIX value.

    Returns
    -------
    bool
    """
    return vix_value is not None and vix_value >= VIX_MODERATE_THRESHOLD


def get_current_vix() -> float | None:
    """
    Simple alias to fetch current VIX value.
    """
    return fetch_vix_price()