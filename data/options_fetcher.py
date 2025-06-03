import os
import requests
from datetime import datetime, timedelta
from utils.logger import bot_logger

TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}


def get_next_expiry(min_dte=2):
    """
    Returns the next Friday expiry with at least `min_dte` days to expiration.
    """
    today = datetime.now().date()
    for i in range(1, 30):
        day = today + timedelta(days=i)
        if day.weekday() == 4 and (day - today).days >= min_dte:
            return day.strftime("%Y-%m-%d")
    raise ValueError("No valid Friday expiry found in next 30 days")


def get_option_chain(symbol="SPY", expiry=None, option_type="call"):
    """
    Fetch full options chain for a given symbol and expiry.
    """
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiry,
        "type": option_type,
        "greeks": "false"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()

    if "options" not in data or data["options"] is None:
        bot_logger.warning(f"[Options Fetch] No contracts returned for {symbol} @ {expiry}")
        return []

    contracts = data["options"].get("option", [])
    return contracts if isinstance(contracts, list) else [contracts]


def get_quote(symbol):
    """
    Fetch real-time quote for a single option contract.
    """
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": symbol}
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()

    quote = data.get("quotes", {}).get("quote")
    return quote if isinstance(quote, dict) else None


def select_moneyness_contracts(contracts, underlying_price, count_per_type=2):
    """
    Selects a mix of ITM, ATM, and OTM contracts closest to the underlying price.
    """
    sorted_contracts = sorted(contracts, key=lambda c: abs(c["strike"] - underlying_price))
    atm_contracts = sorted_contracts[:count_per_type]

    otm_contracts = [c for c in contracts if c["strike"] > underlying_price]
    otm_contracts = sorted(otm_contracts, key=lambda c: c["strike"])[:count_per_type]

    itm_contracts = [c for c in contracts if c["strike"] < underlying_price]
    itm_contracts = sorted(itm_contracts, key=lambda c: c["strike"], reverse=True)[:count_per_type]

    return atm_contracts + otm_contracts + itm_contracts


def fetch_options_bundle(symbol="SPY", min_dte=2, per_moneyness=2):
    """
    High-level fetcher: Gets combined call and put contracts for next expiry with quotes.
    """
    try:
        expiry = get_next_expiry(min_dte=min_dte)

        # Get current underlying price
        quote = get_quote(symbol)
        if not quote:
            bot_logger.warning("[Options Fetch] Failed to get underlying quote")
            return []

        underlying_price = float(quote.get("last", 0))

        contracts = []
        for opt_type in ["call", "put"]:
            chain = get_option_chain(symbol=symbol, expiry=expiry, option_type=opt_type)
            selected = select_moneyness_contracts(chain, underlying_price, count_per_type=per_moneyness)

            # Enrich with quote
            for contract in selected:
                option_quote = get_quote(contract["symbol"])
                if option_quote:
                    contract["quote"] = option_quote
                    contract["option_type"] = opt_type
                    contract["expiry"] = expiry
                    contracts.append(contract)

        bot_logger.info(f"[Options Fetch] Retrieved {len(contracts)} total contracts for {symbol}")
        return contracts

    except Exception as e:
        bot_logger.exception(f"[Options Fetch Error] {e}")
        return []