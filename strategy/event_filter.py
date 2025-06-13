# strategy/event_filter.py

import datetime
import pytz
from utils.logger import bot_logger as logger
from utils.economic_calendar import is_blackout_day, week_has_fomc_or_cpi

eastern = pytz.timezone("US/Eastern")

FED_SPEECH_KEYWORDS = ["Fed", "FOMC", "Federal Reserve", "Powell", "Treasury"]

def is_blackout_time(now=None):
    """
    Checks if the current date is a blackout day due to a high-impact economic event.
    """
    if now is None:
        now = datetime.datetime.now(eastern)
    is_blackout = is_blackout_day(now.date())
    if is_blackout:
        logger.warning(f"🛑 Blackout active — High-impact US economic event on {now.date()}")
    return is_blackout, str(now.date()) if is_blackout else None


def is_fed_event_today(now=None):
    """
    Check if a Fed-related event is scheduled for today.
    """
    if now is None:
        now = datetime.datetime.now(eastern)
    events = week_has_fomc_or_cpi()
    if events:
        logger.warning("📢 Fed-related event (FOMC or CPI) this week — caution advised.")
        return True, "FOMC/CPI this week"
    return False, None


def is_high_risk_event_active(now=None):
    """
    Combined utility to check if today is a blackout day or if a Fed-related event is in the week.
    """
    blackout_active, _ = is_blackout_time(now)
    if blackout_active:
        return True
    fed_event_active, _ = is_fed_event_today(now)
    if fed_event_active:
        return True
    return False


def has_major_event_on(target_date):
    """
    Returns True if target_date is a high-impact US economic event day.
    """
    return is_blackout_day(target_date)