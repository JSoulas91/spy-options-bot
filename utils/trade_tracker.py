# utils/trade_tracker.py

import json
import os
from datetime import datetime, timedelta
from utils.logger import bot_logger

TRADE_LOG_FILE = "data/day_trade_log.json"
MAX_OPEN_TRADES = 8  # Total across day and swing trades

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
            return datetime.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return self._now()

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

    def get_today_day_trade_count(self):
        today_str = self._now().strftime("%Y-%m-%d")
        return sum(
            1 for t in self.trade_log
            if t["type"] == "day" and t["timestamp"].startswith(today_str)
        )

    def get_total_open_trades(self):
        return sum(1 for t in self.trade_log if t["status"] == "open")

    def can_place_trade(self):
        open_count = self.get_total_open_trades()
        if open_count >= MAX_OPEN_TRADES:
            bot_logger.info(f"🚫 Max open trades reached: {open_count}/{MAX_OPEN_TRADES}")
            return False
        return True

    def purge_old_trades(self):
        now = self._now()
        five_days_ago = now - timedelta(days=7)
        original_count = len(self.trade_log)

        self.trade_log = [
            t for t in self.trade_log
            if not (
                (t["type"] == "day" and self._parse_timestamp(t["timestamp"]) < five_days_ago)
                or (t["type"] == "swing" and t["status"] == "closed")
            )
        ]

        removed = original_count - len(self.trade_log)
        if removed > 0:
            bot_logger.info(f"🧹 Purged {removed} old/closed trades from log.")
            self.save_log()

    def log_trade(self, order, trade_type):
        trade_id = self.add_trade("day" if trade_type == 0 else "swing")
        order["id"] = trade_id