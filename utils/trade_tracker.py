import json
import os
from datetime import datetime, date
from threading import Lock
from config import MAX_OPEN_TRADES
from utils.logger import bot_logger

class TradeTracker:
    def __init__(self):
        """
        Initializes the TradeTracker by loading existing trades from the JSON file,
        or starting with an empty list if none exist.
        """
        self.lock = Lock()
        self.trade_log_path = os.path.join(os.path.dirname(__file__), "trade_log.json")
        self.trades = []
        self.load_trades()

    def load_trades(self):
        """
        Loads trades from the trade log file and cleans up old closed trades.
        """
        with self.lock:
            if os.path.exists(self.trade_log_path):
                try:
                    with open(self.trade_log_path, "r") as f:
                        self.trades = json.load(f)
                    self.cleanup_old_trades()
                    bot_logger.info(f"[TradeTracker] Loaded {len(self.trades)} trades from log")
                except Exception as e:
                    bot_logger.error(f"[TradeTracker] Failed to load trades: {e}")
                    self.trades = []
            else:
                self.trades = []

    def save_trades(self):
        """
        Saves current trades to the trade log file atomically to prevent file corruption.
        """
        with self.lock:
            try:
                temp_path = self.trade_log_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(self.trades, f, indent=2)
                os.replace(temp_path, self.trade_log_path)
                bot_logger.debug(f"[TradeTracker] Saved {len(self.trades)} trades to log")
            except Exception as e:
                bot_logger.error(f"[TradeTracker] Failed to save trades: {e}")

    def cleanup_old_trades(self):
        """
        Removes closed trades older than today from the trade list to keep the log manageable.
        """
        with self.lock:
            today_str = date.today().isoformat()
            before_cleanup = len(self.trades)
            # Keep trades that are open or closed today
            self.trades = [
                t for t in self.trades 
                if t.get("status") != "closed" or t.get("close_date") == today_str
            ]
            after_cleanup = len(self.trades)
            if before_cleanup != after_cleanup:
                bot_logger.info(f"[TradeTracker] Cleaned up {before_cleanup - after_cleanup} old closed trades")
                self.save_trades()

    def get_open_trades_count(self):
        """
        Returns the count of currently open trades.
        """
        with self.lock:
            return sum(1 for t in self.trades if t.get("status") == "open")

    def can_place_trade(self):
        """
        Returns True if a new trade can be placed without exceeding MAX_OPEN_TRADES.
        """
        open_count = self.get_open_trades_count()
        if open_count >= MAX_OPEN_TRADES:
            bot_logger.info(f"[TradeTracker] Reached max open trades limit ({MAX_OPEN_TRADES})")
            return False
        return True

    def log_trade(self, trade_data, trade_type):
        """
        Logs a new trade if it doesn't already exist.
        """
        with self.lock:
            trade_id = trade_data.get("id")
            if trade_id is None:
                bot_logger.error("[TradeTracker] Trade data missing 'id', cannot log trade")
                return
            if any(t.get("id") == trade_id and t.get("status") == "open" for t in self.trades):
                bot_logger.warning(f"[TradeTracker] Trade with id {trade_id} is already open, skipping log")
                return

            new_trade = {
                "id": trade_id,
                "symbol": trade_data.get("symbol", ""),
                "timestamp": trade_data.get("timestamp", datetime.utcnow().isoformat()),
                "trade_type": trade_type,
                "status": "open",
                "open_date": date.today().isoformat()
            }
            self.trades.append(new_trade)
            self.save_trades()
            bot_logger.info(f"[TradeTracker] Logged new trade {trade_id} ({trade_type})")

    def close_trade(self, trade_id):
        """
        Marks a trade as closed by its id, setting the close date to today.
        """
        with self.lock:
            for trade in self.trades:
                if trade.get("id") == trade_id and trade.get("status") == "open":
                    trade["status"] = "closed"
                    trade["close_date"] = date.today().isoformat()
                    self.save_trades()
                    bot_logger.info(f"[TradeTracker] Closed trade {trade_id}")
                    return
            bot_logger.warning(f"[TradeTracker] Could not find open trade with id {trade_id} to close")

# Singleton instance for use in other modules
trade_tracker = TradeTracker()