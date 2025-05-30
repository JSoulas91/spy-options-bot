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


def is_fed_speech_active(now=None, lookahead_minutes=30):
    """
    Check if a Fed-related speech is happening now or within the next N minutes.
    """
    if now is None:
        now = datetime.datetime.now(eastern)

    lookahead_time = now + datetime.timedelta(minutes=lookahead_minutes)

    for event in ECONOMIC_EVENTS:
        event_name = event["name"].lower()
        if any(keyword.lower() in event_name for keyword in FED_SPEECH_KEYWORDS):
            start = eastern.localize(datetime.datetime.combine(now.date(), event["start"]))
            end = eastern.localize(datetime.datetime.combine(now.date(), event["end"]))

            if now <= start <= lookahead_time or start <= now <= end:
                logger.warning(f"⚠️ Fed speech active or imminent: {event['name']}")
                return True, event["name"]
    return False, None