# live_runner.py
"""
Real‑time 30‑second loop for the SPY options bot.

* Pulls the full Tradier multi‑timeframe feature set with
  `fetch_long_term_features()` (1‑min, 5‑min, 15‑min, 1‑hr, etc.).
* Runs exit logic first (so we don’t miss a take‑profit / stop‑loss),
  then evaluates / enters new trades via the strategy layer.
* Never dies: any exception is logged and reported to Telegram,
  then the loop keeps going.
"""

import time
import traceback
from datetime import datetime, timezone

from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from data.multi_timeframe_fetcher import fetch_long_term_features

# ── import your own trading logic ──
# If your entry/exit helpers have different names,
# just tweak these two lines.
from trading.exit import handle_exit
from trading.entry import handle_entry
from strategy.strategy import evaluate_trade   # optional, see below


def live_trading_loop(symbol: str = "SPY", interval_sec: int = 30) -> None:
    """
    Runs forever in a daemon thread (started from main.py).
    Fetches data, manages exits, and evaluates new entries twice a minute.
    """
    logger.info(
        f"[LiveRunner] Loop started for {symbol} "
        f"({interval_sec}s cadence)."
    )

    while True:
        cycle_start = time.time()
        try:
            # ─────────────────────────────────────────────────────────
            # 1️⃣  Pull the latest multi‑timeframe features
            # ─────────────────────────────────────────────────────────
            features = fetch_long_term_features(symbol)

            # Grab the most‑recent 1‑minute bar so entry/exit helpers
            # have an “at‑a‑glance” price if they need it.
            one_min = features.get("1min_5d", {})
            last_price = one_min.get("price")

            # Package a minimalist market snapshot for convenience.
            market_snapshot = {
                "symbol": symbol,
                "price": last_price,
                "timestamp": datetime.now(tz=timezone.utc),
                "features": features,
            }

            # ─────────────────────────────────────────────────────────
            # 2️⃣  Manage any open positions first (reduces risk)
            # ─────────────────────────────────────────────────────────
            handle_exit(market_snapshot)

            # ─────────────────────────────────────────────────────────
            # 3️⃣  Evaluate new opportunities / enter trades
            # ─────────────────────────────────────────────────────────
            # If your entry helper already does its own evaluation,
            # comment out the line below and keep only `handle_entry`.
            evaluate_trade(market_snapshot)
            handle_entry(market_snapshot)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"[LiveRunner] {exc}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                f"⚠️ *Live loop error*\n```{str(exc)}```"
            )

        # ─────────────────────────────────────────────────────────
        # 4️⃣  Sleep exactly long enough to keep a stable cadence
        # ─────────────────────────────────────────────────────────
        elapsed = time.time() - cycle_start
        time.sleep(max(0.0, interval_sec - elapsed))
