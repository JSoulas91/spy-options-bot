from datetime import datetime, timedelta

def is_market_open():
    now = datetime.now()
    return now.weekday() < 5 and 9 <= now.hour < 16

def is_pre_market():
    now = datetime.now()
    return now.weekday() < 5 and 4 <= now.hour < 9

def is_after_hours():
    now = datetime.now()
    return now.weekday() < 5 and 16 <= now.hour < 20
