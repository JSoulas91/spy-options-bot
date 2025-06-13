# utils/economic_calendar.py

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional

# In-memory cache
_cached_events = None
_cache_timestamp = None
CACHE_TTL_MINUTES = 60


def get_upcoming_week_dates():
    """Return list of dates (datetime.date) for the next Monday through Friday."""
    today = datetime.utcnow().date()
    days_until_monday = (7 - today.weekday()) % 7
    monday = today + timedelta(days=days_until_monday)
    return [monday + timedelta(days=i) for i in range(5)]


def scrape_forexfactory_calendar():
    """
    Scrape ForexFactory calendar and return list of economic events.
    Each event is a dict with: date, impact, country, event_name.
    """
    global _cached_events, _cache_timestamp

    # Check if cache is valid
    now = datetime.utcnow()
    if _cached_events and _cache_timestamp and (now - _cache_timestamp < timedelta(minutes=CACHE_TTL_MINUTES)):
        return _cached_events

    url = "https://www.forexfactory.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    current_date = None
    rows = soup.select("tr.calendar__row")

    for row in rows:
        date_cell = row.find("td", class_="calendar__date")
        if date_cell and date_cell.get_text(strip=True):
            raw_date = date_cell.get_text(strip=True)
            try:
                current_date = datetime.strptime(raw_date + f".{datetime.utcnow().year}", "%b %d.%Y").date()
            except:
                current_date = None
        if current_date is None:
            continue

        impact_cell = row.find("td", class_="impact")
        impact_level = "low"
        if impact_cell:
            classes = impact_cell.get("class", [])
            if "red" in classes:
                impact_level = "high"
            elif "orange" in classes:
                impact_level = "medium"
            elif "yellow" in classes:
                impact_level = "low"

        country_cell = row.find("td", class_="flag")
        country_code = ""
        if country_cell:
            for c in country_cell.get("class", []):
                if len(c) == 2:
                    country_code = c.lower()
                    break

        event_cell = row.find("td", class_="calendar__event")
        event_name = event_cell.get_text(strip=True) if event_cell else ""

        events.append({
            "date": current_date,
            "impact": impact_level,
            "country": country_code,
            "event_name": event_name,
        })

    _cached_events = events
    _cache_timestamp = now
    return events


def get_high_impact_us_events_for_week():
    """Return sorted list of dates with high-impact US events this week."""
    week_dates = get_upcoming_week_dates()
    events = scrape_forexfactory_calendar()
    blackout_dates = {
        event["date"]
        for event in events
        if event["date"] in week_dates and event["country"] == "us" and event["impact"] == "high"
    }
    return sorted(blackout_dates)


def week_has_fomc_or_cpi():
    """Returns True if the upcoming week includes FOMC or CPI-related high-impact US events."""
    week_dates = get_upcoming_week_dates()
    events = scrape_forexfactory_calendar()
    keywords = ["FOMC", "Federal Reserve", "CPI"]

    for event in events:
        if event["date"] in week_dates and event["country"] == "us" and event["impact"] == "high":
            for kw in keywords:
                if kw.lower() in event["event_name"].lower():
                    return True
    return False


def is_blackout_day(date_to_check: Optional[datetime.date] = None) -> bool:
    """Returns True if the given date (or today) is a high-impact US economic event day."""
    if date_to_check is None:
        date_to_check = datetime.utcnow().date()
    events = scrape_forexfactory_calendar()
    return any(
        event["date"] == date_to_check and event["country"] == "us" and event["impact"] == "high"
        for event in events
    )