"""
Real‑time 30‑second loop for the SPY options bot.

* Pulls multi‑timeframe feature set with `fetch_long_term_features()`
  (1-min, 5-min, 15-min, 1-hr, etc.).
* Runs exit logic first, then entry logic.
* Includes Tradier API throttling and in-memory caching to avoid call overuse.
* Never dies: exceptions are logged, reported via Telegram, and the loop continues.
"""

import time
import traceback
from datetime import datetime, timezone
from threading import Lock

from utils.logger import bot_logger as logger
from utils.telegram import send_telegram_message
from data.multi_timeframe_fetcher import fetch_long_term_features
from monitor.health_check import update_status

# ── Trading logic ──
from trading.exit import handle_exit
from trading.entry import handle_entry
from strategy.strategy import evaluate_trade

# ── Tradier API call throttling + caching ──
MAX_CALLS_PER_MINUTE = 60
CACHE_DURATION = 15  # seconds
crash_log_path = "logs/crash.log"

_last_api_call = 0.0
_api_lock = Lock()

_cached_features = None
_cached_timestamp = 0.0


def throttle_api_calls():
    """Ensure we don’t exceed Tradier free-tier call limits (~1 call/sec)."""
    global _last_api_call
    with _api_lock:
        now = time.time()
        since_last = now - _last_api_call
        if since_last < 1.0:
            time.sleep(1.0 - since_last)
        _last_api_call = time.time()


def get_cached_or_fresh_features(symbol: str) -> dict:
    """Cache Tradier features for a short window to avoid re-fetching."""
    global _cached_features, _cached_timestamp
    now = time.time()
    if _cached_features and (now - _cached_timestamp) < CACHE_DURATION:
        return _cached_features
    throttle_api_calls()
    _cached_features = fetch_long_term_features(symbol)
    _cached_timestamp = now
    return _cached_features


def live_trading_loop(symbol: str = "SPY", interval_sec: int = 30) -> None:
    """
    Runs forever in a daemon thread (started from main.py).
    Evaluates exits and entries with feature caching + API throttling.
    """
    logger.info(
        f"[LiveRunner] Loop started for {symbol} "
        f"({interval_sec}s cadence)."
    )

    while True:
        cycle_start = time.time()
        try:
            update_status("last_loop")

            # ─────────────────────────────────────────────────────────
            # 1️⃣ Pull multi-timeframe features (with caching)
            # ─────────────────────────────────────────────────────────
            features = get_cached_or_fresh_features(symbol)
            one_min = features.get("1min_5d", {})
            last_price = one_min.get("price")

            market_snapshot = {
                "symbol": symbol,
                "price": last_price,
                "timestamp": datetime.now(tz=timezone.utc),
                "features": features,
            }

            # ─────────────────────────────────────────────────────────
            # 2️⃣ Run exit logic (before entry)
            # ─────────────────────────────────────────────────────────
            handle_exit(market_snapshot)

            # ─────────────────────────────────────────────────────────
            # 3️⃣ Run entry evaluation and trade logic
            # ─────────────────────────────────────────────────────────
            evaluate_trade(market_snapshot)
            handle_entry(market_snapshot)

        except Exception as exc:
            logger.error(f"[LiveRunner] {exc}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                f"⚠️ *Live loop error*\n```{str(exc)}```\n"
                f"📄 Crash log: `{crash_log_path}`"
            )

        # ─────────────────────────────────────────────────────────
        # 4️⃣ Sleep to preserve interval cadence
        # ─────────────────────────────────────────────────────────
        elapsed = time.time() - cycle_start
        time.sleep(max(0.0, interval_sec - elapsed))