# live_runner.py
import os, sys, time, traceback, threading
from datetime import datetime, timezone
from functools import wraps
from threading import Lock
from collections import deque
from dotenv import load_dotenv
load_dotenv()

from utils.logger            import bot_logger as logger
from utils.telegram_utils    import send_telegram_message
from monitor.health_check    import update_status
from utils.trade_tracker     import trade_tracker, purge_old_trades
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.quote_utils        import get_spy_quote as original_get_spy_quote
from trading.exit            import handle_exit
from trading.entry           import handle_entry
from strategy.strategy       import evaluate_trade
from meta.online_meta_update import online_update

# ──────────────────────────── Constants / Globals
_last_call = 0.0
_lock = Lock()
API_MIN_DELAY = 1.0
FEATURE_TTL = 15
loop_times = deque(maxlen=100)
crash_path = "logs/crash.log"
_cached_feat, _cached_ts = None, 0.0

# ──────────────────────────── Retry + Throttle
def retry_with_backoff(max_attempts=5, initial_wait=1.0, backoff_factor=2.0, exceptions=(Exception,)):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            wait = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"[Retry] {fn.__name__} failed after {attempt} tries: {e}")
                        raise
                    logger.warning(f"[Retry] {fn.__name__} error: {e} – retry in {wait:.1f}s ({attempt}/{max_attempts})")
                    time.sleep(wait)
                    wait *= backoff_factor
        return wrapper
    return deco

def throttle():
    global _last_call
    with _lock:
        now = time.time()
        if now - _last_call < API_MIN_DELAY:
            time.sleep(API_MIN_DELAY - (now - _last_call))
        _last_call = time.time()

@retry_with_backoff()
def fetch_features_with_retry(symbol: str):
    throttle()
    return fetch_long_term_features(symbol)

@retry_with_backoff()
def get_spy_quote_with_retry():
    return original_get_spy_quote()

def get_features(symbol: str):
    global _cached_feat, _cached_ts
    now = time.time()
    if _cached_feat and now - _cached_ts < FEATURE_TTL:
        return _cached_feat
    _cached_feat, _cached_ts = fetch_features_with_retry(symbol), now
    return _cached_feat

# ──────────────────────────── Meta-agent background update
def schedule_online_update(interval_hours: int = 24):
    def worker():
        while True:
            logger.info("[MetaScheduler] Running online PPO update …")
            try:
                online_update()
                send_telegram_message("🧠 Meta‑agent online update completed.")
            except Exception as e:
                logger.error(f"[MetaScheduler] Update failed: {e}")
                send_telegram_message(f"⚠️ Meta‑agent update failed: {e}")
            logger.info(f"[MetaScheduler] Sleeping {interval_hours} h …")
            time.sleep(interval_hours * 3600)

    threading.Thread(target=worker, daemon=True, name="MetaUpdateScheduler").start()

# ──────────────────────────── Live trading loop
def live_trading_loop(symbol: str = "SPY", interval_sec: int = 20):
    logger.info(f"[LiveRunner] Started for {symbol}, {interval_sec}s cadence")
    try:
        send_telegram_message("▶️ Bot live (20‑sec loop)")
    except Exception as tg_err:
        logger.warning(f"[LiveRunner] Telegram start message failed: {tg_err}")

    schedule_online_update(interval_hours=24)
    kill_switch_path = "/home/ubuntu/kill_switch.txt"

    while True:
        tic = time.time()

        if os.path.exists(kill_switch_path):
            logger.warning("[LiveRunner] Kill‑switch file detected → shutting down")
            try: send_telegram_message("⏹️ Bot stopped via kill‑switch.")
            except: pass
            sys.exit(0)

        try:
            update_status("heartbeat")
            features   = get_features(symbol)
            last_price = get_spy_quote_with_retry()

            snapshot = {
                "symbol":    symbol,
                "price":     last_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features":  features,
            }

            handle_exit(snapshot)
            for tr in trade_tracker.get_open_trades():
                evaluate_trade(tr, snapshot)
            handle_entry(snapshot)

        except Exception as e:
            logger.error(f"[LiveRunner] {e}")
            logger.debug(traceback.format_exc())
            try:
                send_telegram_message(f"⚠️ Loop error\n```{e}```\nLog: {crash_path}")
            except: pass

        loop_times.append(time.time() - tic)
        time.sleep(max(0.0, interval_sec - (time.time() - tic)))

# ──────────────────────────── Unified main (merged from main.py)
def main():
    logger.info("🚀 SPY Options Trading Bot is launching…")
    send_telegram_message("🚀 *Bot Online*\nLive trading loop is starting.")
    purge_old_trades()                          # From main.py
    live_trading_loop()                         # Replaces thread-based launch

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical(f"💀 [Startup Fatal Error] {exc}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"💥 *Fatal Error in startup*\n\n```{exc}```")