# data/tradier_api.py

import os
import requests
from utils.logger import bot_logger as logger

TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID")
USE_LIVE_TRADIER = os.getenv("USE_LIVE_TRADIER", "false").lower() == "true"

BASE_URL = "https://api.tradier.com/v1" if USE_LIVE_TRADIER else "https://sandbox.tradier.com/v1"

HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

# --- Core Helper ---
def safe_request(method, endpoint, params=None, data=None):
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.request(method, url, headers=HEADERS, params=params, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"❌ Tradier API error: {e} | Endpoint: {endpoint}")
        return {}

# --- Quotes ---
def get_equity_quote(symbol="SPY"):
    data = safe_request("GET", "markets/quotes", params={"symbols": symbol})
    return data.get("quotes", {}).get("quote", {})

def get_option_quote(option_symbol):
    data = safe_request("GET", "markets/quotes", params={"symbols": option_symbol})
    return data.get("quotes", {}).get("quote", {})

# --- Option Chain ---
def get_option_chain(symbol="SPY", expiration=None, greeks=False):
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": "true" if greeks else "false"
    }
    data = safe_request("GET", "markets/options/chains", params=params)
    return data.get("options", {}).get("option", [])

# --- Order Placement ---
def place_option_order(option_symbol, quantity, side, order_type="market", duration="day"):
    endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders"
    payload = {
        "class": "option",
        "symbol": "SPY",  # Tradier still expects equity symbol here
        "option_symbol": option_symbol,
        "side": side,  # "buy_to_open", "sell_to_close", etc.
        "quantity": quantity,
        "type": order_type,
        "duration": duration
    }
    data = safe_request("POST", endpoint, data=payload)
    return data.get("order", {})

# --- Order Status & Cancel ---
def check_order_status(order_id):
    endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders/{order_id}"
    data = safe_request("GET", endpoint)
    return data.get("order", {})

def cancel_order(order_id):
    endpoint = f"accounts/{TRADIER_ACCOUNT_ID}/orders/{order_id}"
    data = safe_request("DELETE", endpoint)
    return data.get("order", {})

# --- Account Data ---
def get_account_balances():
    data = safe_request("GET", f"accounts/{TRADIER_ACCOUNT_ID}/balances")
    return data.get("balances", {})

def get_account_positions():
    data = safe_request("GET", f"accounts/{TRADIER_ACCOUNT_ID}/positions")
    return data.get("positions", {}).get("position", [])