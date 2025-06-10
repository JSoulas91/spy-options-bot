# live_runner.py — unified real‑time loop + scheduler helpers
import os
import sys
import time
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from functools   import wraps
from threading   import Lock
from dotenv      import load_dotenv

load_dotenv()  # load environment early

from utils.logger         import bot_logger as logger
from utils.telegram_utils import send_telegram_message
from monitor.health_check import update_status
from utils.trade_tracker  import trade_tracker, purge_old_trades

from data.multi_timeframe_fetcher import fetch_long_term_features
from data.quote_utils             import get_spy_quote as fetch_spy_quote

from trading.exit            import handle_exit
from trading.entry           import handle_entry
from strategy.strategy       import evaluate_trade
from meta.online_meta_update import online_update

# ───────────────────────────────────────────────
# Globals / Throttling
_last_call      = 0.0
_lock           = Lock()
API_MIN_DELAY   = 1.0      # ≥1 s between raw Tradier calls
FEATURE_TTL_SEC = 15

_cached_feat, _cached_ts = None, 0.0
loop_times = deque(maxlen=100)
crash_path = "logs/crash.log"

# ───────────────────────────────────────────────
# Helper decorators
def retry_with_backoff(max_attempts=5, initial_wait=1.0, backoff_factor=2.0, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            wait = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(f"[Retry] {fn.__name__} failed after {attempt} tries: {exc}")
                        raise
                    logger.warning(f"[Retry] {fn.__name__} error: {exc} — retry in {wait:.1f}s "
                                   f"({attempt}/{max_attempts})")
                    time.sleep(wait)
                    wait *= backoff_factor
        return wrapper
    return decorator

def _throttle():
    global _last_call
    with _lock:
        now = time.time()
        if now - _last_call < API_MIN_DELAY:
            time.sleep(API_MIN_DELAY - (now - _last_call))
        _last_call = time.time()

# ───────────────────────────────────────────────
# Data helpers
@retry_with_backoff()
def _fetch_features(symbol: str):
    _throttle()
    return fetch_long_term_features(symbol)

@retry_with_backoff()
def _fetch_spy_quote():
    return fetch_spy_quote()

def _get_features(symbol: str):
    global _cached_feat, _cached_ts
    now = time.time()
    if _cached_feat and now - _cached_ts < FEATURE_TTL_SEC:
        return _cached_feat
    _cached_feat, _cached_ts = _fetch_features(symbol), now
    return _cached_feat

# ───────────────────────────────────────────────
# Background online‑update for the meta‑agent
def _schedule_online_update(interval_hours: int = 24):
    def worker():
        while True:
            logger.info("[MetaScheduler] Running online PPO update …")
            try:
                online_update()
                send_telegram_message("🧠 Meta‑agent online update completed.")
            except Exception as exc:
                logger.error(f"[MetaScheduler] Update failed: {exc}")
                send_telegram_message(f"⚠️ Meta‑agent update failed: {exc}")
            logger.info(f"[MetaScheduler] Sleeping {interval_hours} h …")
            time.sleep(interval_hours * 3600)

    threading.Thread(target=worker, daemon=True, name="MetaUpdateScheduler").start()

# ───────────────────────────────────────────────
# Market‑open housekeeping (callable from scheduler)
def run_market_open_tasks() -> None:
    """
    House‑keeping tasks that should run once at market open
    (or immediately on bot start if already within market hours).
    """
    logger.info("🕘 Running market‑open housekeeping …")
    purge_old_trades()

# ───────────────────────────────────────────────
# Live trading loop
def live_trading_loop(symbol: str = "SPY", interval_sec: int = 20):
    logger.info(f"[LiveRunner] Started for {symbol}, {interval_sec}s cadence")
    try:
        send_telegram_message("▶️ Bot live (20‑sec loop)")
    except Exception as tg_err:
        logger.warning(f"[LiveRunner] Telegram start message failed: {tg_err}")

    _schedule_online_update()  # once, in background
    kill_switch_path = "/home/ubuntu/kill_switch.txt"

    while True:
        start_ts = time.time()

        # Kill‑switch file check
        if os.path.exists(kill_switch_path):
            logger.warning("[LiveRunner] Kill‑switch detected → shutdown")
            try:
                send_telegram_message("⏹️ Bot stopped via kill‑switch.")
            finally:
                sys.exit(0)

        try:
            update_status("heartbeat")            # health‑ping

            features   = _get_features(symbol)
            last_price = _fetch_spy_quote()

            snapshot = {
                "symbol":    symbol,
                "price":     last_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features":  features,
            }

            # 1️⃣ exit management first
            handle_exit(snapshot)

            # 2️⃣ evaluate each open position
            for tr in trade_tracker.get_open_trades():
                evaluate_trade(tr, snapshot)

            # 3️⃣ potential new entry
            handle_entry(snapshot)

        except Exception as exc:
            logger.error(f"[LiveRunner] {exc}")
            logger.debug(traceback.format_exc())
            try:
                send_telegram_message(f"⚠️ Loop error\n```{exc}```\nLog: {crash_path}")
            except:  # noqa: E722
                pass

        # pacing
        elapsed = time.time() - start_ts
        loop_times.append(elapsed)
        time.sleep(max(0.0, interval_sec - elapsed))

# ───────────────────────────────────────────────
# Single‑file entry‑point (merges responsibilities of old main.py)
def main():
    logger.info("🚀 SPY Options Trading Bot is launching …")
    send_telegram_message("🚀 *Bot Online*\nLive trading loop starting.")
    run_market_open_tasks()
    live_trading_loop()  # blocks forever

# ───────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical(f"💀 [Startup Fatal Error] {exc}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"💥 *Fatal Error in startup*\n\n```{exc}```")