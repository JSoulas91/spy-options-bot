# trading/exit.py
import json, random, time, traceback
from datetime import datetime

import pytz
from utils.logger            import bot_logger
from utils.telegram_utils    import send_telegram_message
from utils.trade_tracker     import trade_tracker
from utils.trade_logger      import log_trade_exit
from trade_manager           import close_trade
from meta.meta_state         import build_meta_state_for_exit
from meta.meta_agent         import evaluate_exit_decision
from meta.reward_shaper      import compute_shaped_reward
from config import (
    META_LOG_PATH, SIMULATION_MODE,
    DEFAULT_SLIPPAGE_BPS, SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS,
    HARD_CLOSE_DAYTRADES_ONLY
)

eastern = pytz.timezone("US/Eastern")

def _simulate_close(ref: float):
    slip = ref * DEFAULT_SLIPPAGE_BPS / 10_000
    fill = ref - slip
    delay = random.randint(SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS)
    time.sleep(delay / 1000)
    return round(fill, 4), delay

def _dte(tr):
    exp = tr.get("expiry")
    if not exp:
        return 999
    return (datetime.strptime(exp, "%Y-%m-%d").date() -
            datetime.now(eastern).date()).days

def should_exit(tr):
    """
    Returns reason string if exit recommended, else None.
    Prioritize meta-agent evaluation, then near expiry exit.
    """
    # Meta-agent-driven exit decision
    if evaluate_exit_decision(tr):
        return "Meta-agent"

    # Near expiry auto-exit for type 1 trades (e.g., calls/puts)
    if tr.get("trade_type") == 1 and _dte(tr) <= 1:
        return "Near expiry"

    return None

def close_and_log_trade(tr, reason: str, ref_price: float):
    """
    Closes a trade (simulated or live), computes reward, logs data,
    updates trade tracker, and sends Telegram notification.
    """
    try:
        if SIMULATION_MODE:
            fill, latency = _simulate_close(ref_price)
        else:
            t0 = time.time()
            close_trade(tr)
            latency = int((time.time() - t0) * 1000)
            fill = ref_price

        reward = compute_shaped_reward(tr)
        next_state = build_meta_state_for_exit(tr)

        tr.update({
            "exit_reason": reason,
            "close_price": fill,
            "latency_ms": latency,
            "shaped_reward": reward,
            "meta_next_state": next_state,
        })

        trade_tracker.mark_trade_closed(tr["id"])
        log_trade_exit(tr)

        # Append experience for meta-agent training
        with open(META_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "state": tr.get("meta_state"),
                "action": tr.get("meta_action"),
                "reward": reward,
                "next_state": next_state,
                "done": True
            }) + "\n")

        send_telegram_message(
            f"🔴 Exit {tr['symbol']} {tr['id']} | {reason} | Reward {reward:.3f}"
        )
    except Exception as exc:
        bot_logger.error(f"[Exit-close] {exc}")
        bot_logger.debug(traceback.format_exc())

def handle_exit(snapshot: dict | None = None):
    """
    Main loop to check all open trades for exit signals.
    Also optionally hard closes only day trades at 15:55 market time.
    """
    try:
        now = datetime.now(eastern)

        # Hard exit at 15:55 to close positions before market close
        if now.hour == 15 and now.minute >= 55:
            for tr in list(trade_tracker.get_open_trades()):
                if HARD_CLOSE_DAYTRADES_ONLY:
                    # Close only day trades if flag set
                    if not tr.get("is_daytrade", False):
                        continue  # skip non-day trades
                close_and_log_trade(tr, "Auto 15:55", snapshot["price"] if snapshot else 0)
            return

        ref_price = snapshot["price"] if snapshot else 0
        for tr in list(trade_tracker.get_open_trades()):
            reason = should_exit(tr)
            if reason:
                close_and_log_trade(tr, reason, ref_price)

    except Exception as exc:
        bot_logger.error(f"[Exit] {exc}")
        bot_logger.debug(traceback.format_exc())