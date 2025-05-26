import requests
from datetime import datetime, timedelta

class EconomicCalendar:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://calendarific.com/api/v2/holidays"

    def get_us_events(self):
        today = datetime.utcnow().date()
        tomorrow = today + timedelta(days=1)

        params = {
            "api_key": self.api_key,
            "country": "US",
            "year": today.year,
            "type": "national"
        }

        try:
            response = requests.get(self.endpoint, params=params)
            data = response.json()
            events = data.get("response", {}).get("holidays", [])

            blackout_dates = []
            for event in events:
                event_date = datetime.strptime(event["date"]["iso"], "%Y-%m-%d").date()
                if today <= event_date <= tomorrow:
                    blackout_dates.append(event_date)

            return blackout_dates

        except Exception as e:
            print(f"[Calendar Error] {e}")
            return []

    def is_blackout_today(self):
        today = datetime.utcnow().date()
        blackout_dates = self.get_us_events()
        return today in blackout_dates
