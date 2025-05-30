# event_filter.py

import datetime
import pytz
from config import ECONOMIC_EVENTS, FED_SPEECH_KEYWORDS
from utils.logger import bot_logger as logger

eastern = pytz.timezone("US/Eastern")

def is_blackout_time(now=None):
    """
    Check if the current time falls within any high-risk economic event blackout window.
    """
    if now is None:
        now = datetime.datetime.now(eastern)

    for event in ECONOMIC_EVENTS:
        start_dt = datetime.datetime.combine(now.date(), event["start"])
        end_dt = datetime.datetime.combine(now.date(), event["end"])
        start = eastern.localize(start_dt)
        end = eastern.localize(end_dt)

        if start <= now <= end:
            logger.warning(f"🛑 Blackout active — Economic Event: {event['name']}")
            return True, event["name"]
    return False, None


def is_fed_event_today(now=None):
    """
    Check if a Fed-related event is scheduled for today.
    """
    if now is None:
        now = datetime.datetime.now(eastern)

    for event in ECONOMIC_EVENTS:
        if event["date"] == now.date():
            if any(keyword.lower() in event["name"].lower() for keyword in FED_SPEECH_KEYWORDS):
                logger.warning(f"📢 Fed Speech Event Today: {event['name']}")
                return True, event["name"]
    return False, None


def is_high_risk_event_active(now=None):
    """
    Combined utility to check if either a blackout or Fed speech is active.
    """
    blackout_active, blackout_name = is_blackout_time(now)
    if blackout_active:
        return True

    fed_event, fed_name = is_fed_event_today(now)
    if fed_event:
        return True

    return False