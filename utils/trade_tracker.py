import json
import os
from datetime import datetime, date
from threading import Lock
from config import MAX_OPEN_TRADES
from utils.logger import bot_logger

class TradeTracker:
    def __init__(self):
        self.lock = Lock()
        self.trade_log_path = os.path.join(os.path.dirname(__file__), "trade_log.json")
        self.trades = []
        self.load_trades()

    def load_trades(self):
        if os.path.exists(self.trade_log_path):
            try:
                with open(self.trade_log_path, "r") as f:
                    self.trades = json.load(f)
                self.cleanup_old_trades()
            except Exception as e:
                bot_logger.error(f"[TradeTracker] Failed to load trades: {e}")
                self.trades = []
        else:
            self.trades = []

    def save_trades(self):
        try:
            with open(self.trade_log_path, "w") as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            bot_logger.error(f"[TradeTracker] Failed to save trades: {e}")

    def cleanup_old_trades(self):
        today_str = date.today().isoformat()
        before_cleanup = len(self.trades)
        self.trades = [t for t in self.trades if t.get("status") != "closed" or t.get("close_date") == today_str]
        after_cleanup = len(self.trades)
        if before_cleanup != after_cleanup:
            bot_logger.info(f"[TradeTracker] Cleaned up {before_cleanup - after_cleanup} old closed trades")
            self.save_trades()

    def get_open_trades_count(self):
        with self.lock:
            open_count = sum(1 for t in self.trades if t.get("status") == "open")
            return open_count

    def can_place_trade(self):
        with self.lock:
            open_count = self.get_open_trades_count()
            if open_count >= MAX_OPEN_TRADES:
                bot_logger.info(f"[TradeTracker] Reached max open trades limit ({MAX_OPEN_TRADES})")
                return False
            return True

    def log_trade(self, trade_data, trade_type):
        with self.lock:
            self.trades.append({
                "id": trade_data.get("id"),
                "symbol": trade_data.get("symbol"),
                "timestamp": trade_data.get("timestamp"),
                "trade_type": trade_type,
                "status": "open"
            })
            self.save_trades()

    def close_trade(self, trade_id):
        with self.lock:
            for trade in self.trades:
                if trade.get("id") == trade_id and trade.get("status") == "open":
                    trade["status"] = "closed"
                    trade["close_date"] = date.today().isoformat()
                    self.save_trades()
                    break

trade_tracker = TradeTracker()