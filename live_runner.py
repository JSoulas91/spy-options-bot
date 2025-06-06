import time
import traceback
import os
import sys
from datetime import datetime, timezone
from threading import Lock, deque
from functools import wraps

from utils.logger import bot_logger as logger
from utils.telegram import send_telegram_message
from monitor.health_check import update_status

from data.multi_timeframe_fetcher import fetch_long_term_features
from data.quote_utils            import get_spy_quote as original_get_spy_quote
from trading.exit     import handle_exit
from trading.entry    import handle_entry
from strategy.strategy import evaluate_trade

# ── Global throttling vars ──
_last_call = 0.0
_lock      = Lock()
API_MIN_DELAY = 1.0  # ≥1 s between raw Tradier calls

# ── Feature cache ──
_cached_feat = None
_cached_ts   = 0.0
FEATURE_TTL  = 15  # seconds

# ── Crash stats ──
crash_path = "logs/crash.log"
loop_times = deque(maxlen=100)

# ── Retry decorator with exponential backoff ──
def retry_with_backoff(max_attempts=5, initial_wait=1.0, backoff_factor=2.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait = initial_wait
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"[Retry] Max retries reached for {func.__name__}: {e}")
                        raise
                    logger.warning(f"[Retry] {func.__name__} failed with {e}, retrying in {wait:.1f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(wait)
                    wait *= backoff_factor
        return wrapper
    return decorator

def throttle():
    """Block until ≥1 s since last Tradier call (simple token bucket)."""
    global _last_call
    with _lock:
        now = time.time()
        if now - _last_call < API_MIN_DELAY:
            time.sleep(API_MIN_DELAY - (now - _last_call))
        _last_call = time.time()

@retry_with_backoff(max_attempts=5, initial_wait=1.0)
def fetch_features_with_retry(symbol: str):
    throttle()
    return fetch_long_term_features(symbol)

@retry_with_backoff(max_attempts=5, initial_wait=1.0)
def get_spy_quote_with_retry():
    return original_get_spy_quote()

def get_features(symbol: str):
    global _cached_feat, _cached_ts
    now = time.time()
    if _cached_feat and now - _cached_ts < FEATURE_TTL:
        return _cached_feat
    _cached_feat = fetch_features_with_retry(symbol)
    _cached_ts   = now
    return _cached_feat

def live_trading_loop(symbol: str = "SPY", interval_sec: int = 20):
    logger.info(f"[LiveRunner] Started for {symbol}, interval {interval_sec}s")
    send_telegram_message("▶️ Bot live (20‑sec loop)")

    kill_switch_path = "/home/ubuntu/kill_switch.txt"

    while True:
        tic = time.time()

        # Kill switch check
        if os.path.exists(kill_switch_path):
            logger.warning("[LiveRunner] Kill switch detected. Stopping bot.")
            send_telegram_message("⏹️ Bot stopping via kill switch.")
            sys.exit(0)

        try:
            update_status("heartbeat")
            features   = get_features(symbol)
            last_price = get_spy_quote_with_retry()  # cached 6 s internally or via bot logic

            snapshot = {
                "symbol":    symbol,
                "price":     last_price,
                "timestamp": datetime.now(timezone.utc),
                "features":  features,
            }

            handle_exit(snapshot)          # risk first
            evaluate_trade(snapshot)       # meta evaluation
            handle_entry(snapshot)         # maybe open

        except Exception as e:
            logger.error(f"[LiveRunner] {e}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                f"⚠️ Loop error\n```{e}```\nLog: {crash_path}"
            )

        loop_times.append(time.time() - tic)
        time.sleep(max(0.0, interval_sec - (time.time() - tic)))