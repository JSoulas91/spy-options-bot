# utils/economic_calendar.py

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_upcoming_week_dates():
    """Return list of dates (datetime.date) for the next Monday through Friday."""
    today = datetime.utcnow().date()
    days_until_monday = (7 - today.weekday()) % 7
    monday = today + timedelta(days=days_until_monday)
    return [monday + timedelta(days=i) for i in range(5)]  # Mon to Fri

def scrape_forexfactory_calendar():
    """
    Scrape ForexFactory calendar page and return a list of events.
    Each event is a dict with keys:
    - date (datetime.date)
    - impact ('low', 'medium', 'high')
    - country (str)
    - event_name (str)
    """
    url = "https://www.forexfactory.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    events = []
    current_date = None

    rows = soup.select("tr.calendar__row")
    for row in rows:
        # Check if row has a date cell — some rows group events by date
        date_cell = row.find("td", class_="calendar__date")
        if date_cell and date_cell.get_text(strip=True):
            # Parse the date (e.g. "Jun 10")
            raw_date = date_cell.get_text(strip=True)
            try:
                current_date = datetime.strptime(raw_date + f".{datetime.utcnow().year}", "%b %d.%Y").date()
            except:
                current_date = None

        # Skip if no valid date found yet
        if current_date is None:
            continue

        # Impact: check cell with class 'impact' that may have 'red', 'orange', or 'yellow'
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

        # Country: check 'flag' class and country code in class list
        country_cell = row.find("td", class_="flag")
        country_code = ""
        if country_cell:
            for c in country_cell.get("class", []):
                if len(c) == 2:
                    country_code = c.lower()
                    break

        # Event name
        event_cell = row.find("td", class_="calendar__event")
        event_name = event_cell.get_text(strip=True) if event_cell else ""

        events.append({
            "date": current_date,
            "impact": impact_level,
            "country": country_code,
            "event_name": event_name,
        })

    return events


def get_high_impact_us_events_for_week():
    """
    Return list of dates for upcoming week (Mon-Fri) that have high-impact US events.
    """
    week_dates = get_upcoming_week_dates()
    events = scrape_forexfactory_calendar()
    blackout_dates = set()

    for event in events:
        if event["date"] in week_dates and event["country"] == "us" and event["impact"] == "high":
            blackout_dates.add(event["date"])

    return sorted(blackout_dates)


def week_has_fomc_or_cpi():
    """
    Checks if upcoming week contains FOMC meeting or CPI release in US calendar.

    Returns True if either event appears as a high-impact event.
    """
    week_dates = get_upcoming_week_dates()
    events = scrape_forexfactory_calendar()
    keywords = ["FOMC", "Federal Reserve", "CPI"]

    for event in events:
        if event["date"] in week_dates and event["country"] == "us" and event["impact"] == "high":
            for kw in keywords:
                if kw.lower() in event["event_name"].lower():
                    return True
    return False


def is_blackout_day(date_to_check: datetime.date | None = None) -> bool:
    """
    Returns True if the provided date (or today if None) has a high-impact US event.

    Parameters
    ----------
    date_to_check : datetime.date | None
        The date to check. Defaults to today (UTC).

    Returns
    -------
    bool
    """
    if date_to_check is None:
        date_to_check = datetime.utcnow().date()
    events = scrape_forexfactory_calendar()
    for event in events:
        if event["date"] == date_to_check and event["country"] == "us" and event["impact"] == "high":
            return True
    return False