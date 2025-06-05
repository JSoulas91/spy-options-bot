# main.py

import time
import traceback
import threading

from scheduler import run_scheduler_loop
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from utils.trade_tracker import purge_old_trades

# ⬇️ NEW: real‑time loop that fetches all Tradier time‑frames twice a minute
from live_runner import live_trading_loop   # create this module if you haven’t already


def run_market_open_tasks() -> None:
    """
    House‑keeping work that should run once at market open
    (or at bot start if already within market hours).
    """
    logger.info("🕘 Running market‑open housekeeping…")
    purge_old_trades()


def main() -> None:
    logger.info("🚀 SPY Options Trading Bot is launching…")
    send_telegram_message(
        "🚀 *Bot Online*\nMain loop is starting and will self‑heal on failure."
    )

    run_market_open_tasks()

    # ──────────────────────────────────────────────────────────────────
    # Start the 30‑second live‑trading loop in its own daemon thread
    # ──────────────────────────────────────────────────────────────────
    live_thread = threading.Thread(
        target=live_trading_loop,      # pulls Tradier data & handles entry/exit
        daemon=True,
        name="LiveTradingLoop",
    )
    live_thread.start()
    logger.info("🔄 Live‑trading loop started (30‑second cadence).")

    # ──────────────────────────────────────────────────────────────────
    # Keep the scheduler (daily summaries, retrains, etc.) running
    # ──────────────────────────────────────────────────────────────────
    while True:
        try:
            run_scheduler_loop()       # this call blocks; should never return
        except Exception as exc:
            logger.critical(f"🔥 [Main Loop Crash] {exc}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                "❌ *Main Loop Crash Detected*\n"
                f"📛 Error: `{exc}`\n"
                "🔁 Retrying in 5 seconds…"
            )
            time.sleep(5)
        else:
            # `run_scheduler_loop()` exited unexpectedly;
            # restart after a short pause to self‑heal.
            logger.warning("⚠️ [Main] Scheduler loop exited unexpectedly.")
            send_telegram_message(
                "⚠️ *Scheduler loop exited unexpectedly.* Retrying in 5 seconds…"
            )
            time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical(f"💀 [Startup Fatal Error] {exc}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"💥 *Fatal Error in `main.py` startup*\n\n```{exc}```"
        )