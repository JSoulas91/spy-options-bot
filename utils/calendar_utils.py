from datetime import datetime
import pandas_market_calendars as mcal

def is_market_open():
    nyse = mcal.get_calendar('NYSE')
    now = datetime.utcnow()
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    return not schedule.empty and schedule.at[schedule.index[0], 'market_open'] <= now <= schedule.at[schedule.index[0], 'market_close']
