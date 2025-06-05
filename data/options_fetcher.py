import os
import time
import requests
from datetime import datetime, timedelta
from functools import lru_cache
from utils.logger import bot_logger

TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

# -------------------- TTL Cache --------------------
cache_store = {}

def ttl_cache(key, ttl=30):
    now = time.time()
    if key in cache_store:
        value, expiry = cache_store[key]
        if now < expiry:
            return value
        else:
            del cache_store[key]
    return None

def set_cache(key, value, ttl=30):
    cache_store[key] = (value, time.time() + ttl)

# -------------------- Utilities --------------------

@lru_cache(maxsize=1)
def get_upcoming_fridays(count=3, min_dte=2):
    """
    Returns the next `count` Friday expiries with at least `min_dte` days until expiration.
    Cached since it doesn't change often.
    """
    today = datetime.now().date()
    fridays = []
    for i in range(1, 45):
        day = today + timedelta(days=i)
        if day.weekday() == 4 and (day - today).days >= min_dte:
            fridays.append(day.strftime("%Y-%m-%d"))
        if len(fridays) >= count:
            break
    return fridays


def get_quote(symbol):
    """
    Fetch a quote for a given OCC-formatted option or equity symbol, with caching.
    """
    cache_key = f"quote:{symbol}"
    cached = ttl_cache(cache_key, ttl=20)  # quotes update every ~10s
    if cached:
        return cached

    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": symbol, "greeks": "true"}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        quote = data.get("quotes", {}).get("quote")
        if isinstance(quote, dict):
            set_cache(cache_key, quote, ttl=20)
            return quote
        return None
    except Exception as e:
        bot_logger.exception(f"[Quote Fetch Error] {e}")
        return None


@lru_cache(maxsize=1)
def get_expiration_dates(symbol="SPY"):
    """
    Cached expiration dates.
    """
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    params = {
        "symbol": symbol,
        "includeAllRoots": "true",
        "strikes": "true"
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()
        dates = data.get("expirations", {}).get("date", [])
        return dates if isinstance(dates, list) else [dates]
    except Exception as e:
        bot_logger.exception(f"[Expirations Fetch Error] {e}")
        return []


def get_option_chain(symbol="SPY", expiry=None, option_type="call"):
    """
    Cached option chain per (symbol, expiry, type).
    """
    cache_key = f"chain:{symbol}:{expiry}:{option_type}"
    cached = ttl_cache(cache_key, ttl=180)  # option chains change less frequently
    if cached:
        return cached

    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiry,
        "type": option_type,
        "greeks": "false"
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()

        if "options" not in data or data["options"] is None:
            bot_logger.warning(f"[Options Fetch] No contracts returned for {symbol} @ {expiry}")
            return []

        contracts = data["options"].get("option", [])
        if not isinstance(contracts, list):
            contracts = [contracts]

        set_cache(cache_key, contracts, ttl=180)
        return contracts

    except Exception as e:
        bot_logger.exception(f"[Option Chain Error] {e}")
        return []


def select_moneyness_contracts(contracts, underlying_price, count_per_type=2):
    sorted_contracts = sorted(contracts, key=lambda c: abs(c["strike"] - underlying_price))
    atm_contracts = sorted_contracts[:count_per_type]

    otm_contracts = [c for c in contracts if c["strike"] > underlying_price]
    otm_contracts = sorted(otm_contracts, key=lambda c: c["strike"])[:count_per_type]

    itm_contracts = [c for c in contracts if c["strike"] < underlying_price]
    itm_contracts = sorted(itm_contracts, key=lambda c: c["strike"], reverse=True)[:count_per_type]

    return atm_contracts + otm_contracts + itm_contracts


def fetch_options_bundle(symbol="SPY", expiries=3, per_moneyness=2):
    """
    Fetch ITM/ATM/OTM contracts across upcoming expirations.
    Uses quote and chain caching to reduce API calls.
    """
    try:
        expiry_dates = get_upcoming_fridays(count=expiries, min_dte=2)
        quote = get_quote(symbol)
        if not quote:
            bot_logger.warning("[Options Fetch] Failed to get underlying quote")
            return []

        underlying_price = float(quote.get("last", 0))
        contracts = []

        for expiry in expiry_dates:
            for opt_type in ["call", "put"]:
                chain = get_option_chain(symbol=symbol, expiry=expiry, option_type=opt_type)
                selected = select_moneyness_contracts(chain, underlying_price, count_per_type=per_moneyness)

                for contract in selected:
                    opt_sym = contract["symbol"]
                    opt_quote = get_quote(opt_sym)  # cached per symbol
                    if opt_quote:
                        contract["quote"] = opt_quote
                        contract["option_type"] = opt_type
                        contract["expiry"] = expiry
                        contracts.append(contract)

        bot_logger.info(f"[Options Fetch] Retrieved {len(contracts)} contracts across {len(expiry_dates)} expiries for {symbol}")
        return contracts

    except Exception as e:
        bot_logger.exception(f"[Options Fetch Error] {e}")
        return []


def get_option_metrics(symbol="SPY"):
    """
    Pulls Greeks and IV from top 5 ATM/OTM call and put options.
    """
    try:
        expiry = get_upcoming_fridays(count=1, min_dte=2)[0]
        quote = get_quote(symbol)
        if not quote:
            return {}

        price = float(quote.get("last", 0))
        metrics = {}

        for opt_type in ["call", "put"]:
            chain = get_option_chain(symbol=symbol, expiry=expiry, option_type=opt_type)
            if not chain:
                continue

            filtered = sorted(chain, key=lambda c: abs(c["strike"] - price))[:5]

            for contract in filtered:
                opt_sym = contract["symbol"]
                opt_quote = get_quote(opt_sym)
                if opt_quote and all(k in opt_quote for k in ["delta", "gamma", "theta", "vega", "rho", "iv"]):
                    metrics[opt_type] = {
                        "symbol": opt_sym,
                        "strike": contract["strike"],
                        "delta": opt_quote["delta"],
                        "gamma": opt_quote["gamma"],
                        "theta": opt_quote["theta"],
                        "vega": opt_quote["vega"],
                        "rho": opt_quote["rho"],
                        "iv": opt_quote["iv"],
                        "price": opt_quote.get("last"),
                        "expiry": expiry
                    }
                    break

        return metrics

    except Exception as e:
        bot_logger.exception(f"[Option Metrics Fetch Error] {e}")
        return {}