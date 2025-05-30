# event_filter.py

import datetime
import pytz
from config import ECONOMIC_EVENTS, FED_SPEECH_KEYWORDS
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
    Check for any Fed-related scheduled speeches today (basic keyword match).
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