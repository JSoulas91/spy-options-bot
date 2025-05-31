# utils/trade_tracker.py

import json
import os
from datetime import datetime
from config import MAX_DAY_TRADES, ENFORCE_PDT_LIMITS
from utils.logger import bot_logger

TRADE_LOG_FILE = "data/day_trade_log.json"


class TradeTracker:
    def __init__(self):
        self.trade_log = {}
        self.load_log()

    def load_log(self):
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, "r") as f:
                self.trade_log = json.load(f)
        else:
            self.trade_log = {}

    def save_log(self):
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(self.trade_log, f, indent=2)

    def _get_today_key(self):
        return datetime.now().strftime("%Y-%m-%d")

    def increment_trade_count(self):
        today = self._get_today_key()
        self.trade_log[today] = self.trade_log.get(today, 0) + 1
        self.save_log()
        bot_logger.info(f"📈 Trade count for {today}: {self.trade_log[today]}")

    def get_today_trade_count(self):
        return self.trade_log.get(self._get_today_key(), 0)

    def can_execute_trade(self):
        if not ENFORCE_PDT_LIMITS:
            return True
        count = self.get_today_trade_count()
        if count < MAX_DAY_TRADES:
            return True
        bot_logger.warning(f"🚫 PDT limit reached: {count} trades today (max = {MAX_DAY_TRADES})")
        return False

    def reset_daily_log_if_new_day(self):
        """Optional: call this at market open to ensure clean log rollover."""
        today = self._get_today_key()
        keys_to_delete = [k for k in self.trade_log if k != today]
        for k in keys_to_delete:
            del self.trade_log[k]
        self.save_log()