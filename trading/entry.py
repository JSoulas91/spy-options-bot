# event_filter.py

import datetime
import pytz
import requests
from config import ECONOMIC_EVENTS, FED_SPEECH_KEYWORDS, VIX_RISK_THRESHOLD
from utils.logger import bot_logger as logger

eastern = pytz.timezone("US/Eastern")

def is_blackout_time(now=None):
    """
    Check if the current time falls within an economic event blackout window.
    """
    if now is None:
        now = datetime.datetime.now(eastern)

    for event in ECONOMIC_EVENTS:
        start = eastern.localize(datetime.datetime.combine(now.date(), event["start"]))
        end = eastern.localize(datetime.datetime.combine(now.date(), event["end"]))
        if start <= now <= end:
            logger.info(f"🛑 Trading blocked due to economic event: {event['name']}")
            return True, event["name"]
    return False, None

def is_fed_event_today(now=None):
    """
    Check for any Fed-related scheduled speeches today.
    """
    if now is None:
        now = datetime.datetime.now(eastern)

    for event in ECONOMIC_EVENTS:
        if event["date"] == now.date() and any(
            keyword.lower() in event["name"].lower() for keyword in FED_SPEECH_KEYWORDS
        ):
            logger.warning(f"📢 Fed-related event detected: {event['name']}")
            return True, event["name"]
    return False, None

def is_vix_high():
    """
    Check if the VIX index is above a danger threshold.
    """
    try:
        # Replace with real API request or mock this in testing
        response = requests.get("https://api.tradier.com/v1/markets/quotes?symbols=VIX",
                                headers={"Authorization": "Bearer YOUR_TOKEN", "Accept": "application/json"})
        data = response.json()
        vix_price = float(data["quotes"]["quote"]["last"])
        if vix_price >= VIX_RISK_THRESHOLD:
            logger.warning(f"⚠️ VIX is elevated at {vix_price} — risk-off environment.")
            return True, vix_price
    except Exception as e:
        logger.error(f"[VIX Check Error] Could not retrieve VIX: {str(e)}")
    return False, None

def is_high_risk_event_active():
    """
    Combine all major event-based risk checks into a single filter.
    """
    blackout, event = is_blackout_time()
    if blackout:
        return True

    fed_event, _ = is_fed_event_today()
    if fed_event:
        return True

    vix_risk, _ = is_vix_high()
    if vix_risk:
        return True

    return False