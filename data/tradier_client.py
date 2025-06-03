# data/tradier_client.py

import requests
import os
from config import TRADIER_ACCESS_TOKEN, USE_TRADIER_SANDBOX
from utils.logger import bot_logger

TRADIER_BASE_URL = "https://sandbox.tradier.com/v1" if USE_TRADIER_SANDBOX else "https://api.tradier.com/v1"

HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json"
}


def get_option_chain(symbol, expiry, option_type="call"):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiry,
        "greeks": "true",
        "type": option_type
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    else:
        bot_logger.error(f"Failed to get option chain: {response.text}")
        return []


def get_option_quote(symbol):
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": symbol}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        quotes = response.json().get("quotes", {}).get("quote")
        if isinstance(quotes, list):
            return quotes
        else:
            return [quotes]
    else:
        bot_logger.error(f"Failed to get option quote: {response.text}")
        return []


def place_option_order(account_id, symbol, quantity, action, order_type="market", duration="day"):
    url = f"{TRADIER_BASE_URL}/accounts/{account_id}/orders"
    payload = {
        "class": "option",
        "symbol": symbol,
        "side": action,
        "quantity": quantity,
        "type": order_type,
        "duration": duration
    }
    response = requests.post(url, headers=HEADERS, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        bot_logger.error(f"Failed to place order: {response.text}")
        return None


def get_account_id():
    url = f"{TRADIER_BASE_URL}/user/profile"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        accounts = response.json().get("profile", {}).get("accounts", {}).get("account", [])
        if isinstance(accounts, list):
            return accounts[0].get("account_number") if accounts else None
        return accounts.get("account_number")
    else:
        bot_logger.error(f"Failed to retrieve account ID: {response.text}")
        return None


def get_market_clock():
    url = f"{TRADIER_BASE_URL}/markets/clock"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("clock", {})
    else:
        bot_logger.error(f"Failed to retrieve market clock: {response.text}")
        return {}