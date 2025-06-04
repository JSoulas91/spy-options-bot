# utils/trade_tracker.py

import json
import os
from datetime import datetime, timedelta
from config import MAX_DAY_TRADES, ENFORCE_PDT_LIMITS
from utils.logger import bot_logger

TRADE_LOG_FILE = "data/day_trade_log.json"

class TradeTracker:
    def __init__(self):
        self.trade_log = []
        self.load_log()

    def load_log(self):
        try:
            if os.path.exists(TRADE_LOG_FILE):
                with open(TRADE_LOG_FILE, "r") as f:
                    self.trade_log = json.load(f)
            else:
                self.trade_log = []
        except Exception as e:
            bot_logger.error(f"❌ Failed to load trade log: {e}")
            self.trade_log = []

    def save_log(self):
        try:
            with open(TRADE_LOG_FILE, "w") as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception as e:
            bot_logger.error(f"❌ Failed to save trade log: {e}")

    def _now(self):
        return datetime.now()

    def _parse_timestamp(self, ts):
        try:
            # Handle ISO8601 and possible UTC suffix
            return datetime.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return self._now()  # Fallback if malformed

    def _new_trade_id(self):
        return self._now().strftime("%Y-%m-%dT%H:%M:%S")

    def add_trade(self, trade_type="day"):
        trade = {
            "id": self._new_trade_id(),
            "timestamp": self._new_trade_id(),
            "type": trade_type,
            "status": "open"
        }
        self.trade_log.append(trade)
        self.save_log()
        bot_logger.info(f"📌 Logged new {trade_type} trade: {trade['id']}")
        return trade["id"]

    def close_trade(self, trade_id):
        for trade in self.trade_log:
            if trade["id"] == trade_id and trade["status"] == "open":
                trade["status"] = "closed"
                bot_logger.info(f"✅ Closed trade: {trade_id}")
                self.save_log()
                return True
        bot_logger.warning(f"⚠️ Tried to close unknown or already closed trade: {trade_id}")
        return False

    def get_open_swing_trades(self):
        return [t for t in self.trade_log if t["type"] == "swing" and t["status"] == "open"]

    def get_recent_day_trades(self):
        now = self._now()
        cutoff = now - timedelta(days=7)  # 5 business days buffer
        return [
            t for t in self.trade_log
            if t["type"] == "day" and self._parse_timestamp(t["timestamp"]) >= cutoff
        ]

    def get_today_day_trade_count(self):
        today_str = self._now().strftime("%Y-%m-%d")
        return sum(
            1 for t in self.get_recent_day_trades()
            if t["timestamp"].startswith(today_str)
        )

    def can_execute_trade(self):
        if not ENFORCE_PDT_LIMITS:
            return True
        count = self.get_today_day_trade_count()
        if count < MAX_DAY_TRADES:
            return True
        bot_logger.warning(f"🚫 PDT limit reached: {count} trades today (max = {MAX_DAY_TRADES})")
        return False

    def purge_old_trades(self):
        now = self._now()
        five_business_days_ago = now - timedelta(days=7)
        original_count = len(self.trade_log)

        self.trade_log = [
            t for t in self.trade_log
            if not (
                (t["type"] == "day" and self._parse_timestamp(t["timestamp"]) < five_business_days_ago)
                or (t["type"] == "swing" and t["status"] == "closed")
            )
        ]

        removed = original_count - len(self.trade_log)
        if removed > 0:
            bot_logger.info(f"🧹 Purged {removed} old/closed trades from log.")
            self.save_log()