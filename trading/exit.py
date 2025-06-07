# trading/exit.py
"""
Handles exit logic.

• In SIMULATION_MODE the close is simulated with slippage & latency
• Otherwise we call Tradier through close_trade()
"""

import json, random, time, traceback
from datetime import datetime

import pytz
from utils.logger           import bot_logger
from utils.trade_logger     import log_trade_exit
from utils.trade_tracker    import trade_tracker
from utils.telegram_notifier import TelegramNotifier
from trade_manager          import close_trade
from meta.meta_agent        import evaluate_exit_decision
from meta.meta_state        import build_meta_state_for_exit
from meta.reward_shaper     import compute_shaped_reward
from config                 import (
    META_LOG_PATH, SIMULATION_MODE,
    DEFAULT_SLIPPAGE_BPS, SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS
)

eastern  = pytz.timezone("US/Eastern")
notifier = TelegramNotifier()


# ──────────────────────────────────────────
def _simulate_close(fill_ref: float, side: str):
    slip = fill_ref * DEFAULT_SLIPPAGE_BPS / 10_000
    price = fill_ref - slip if side == "sell" else fill_ref + slip
    delay = random.randint(SIM_MIN_FILL_DELAY_MS, SIM_MAX_FILL_DELAY_MS)
    time.sleep(delay / 1000)
    return round(price, 4), delay


def handle_exit(market_snapshot=None):
    try:
        now = datetime.now(eastern)

        # Auto close day‑trades 3:55 PM
        if now.hour == 15 and now.minute >= 55:
            for tr in list(trade_tracker.get_open_trades()):
                close_and_log_trade(tr, "Auto 15:55 exit")
            return

        for tr in list(trade_tracker.get_open_trades()):
            reason = should_exit_trade(tr)
            if reason:
                close_and_log_trade(tr, reason, ref_price=market_snapshot["price"] if market_snapshot else None)

    except Exception as exc:
        bot_logger.exception(f"[EXIT] {exc}")


def should_exit_trade(trade: dict):
    try:
        if evaluate_exit_decision(trade):
            return "Meta‑agent"
        if trade.get("trade_type") == 1 and _dte(trade) <= 1:
            return "Near expiry"
        return None
    except Exception as e:
        bot_logger.error(f"[Exit‑eval] {e}")
        return None


def _dte(tr):
    exp = tr.get("expiry");  # yyyy‑mm‑dd
    if not exp: return 999
    return (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now(eastern).date()).days


def close_and_log_trade(trade: dict, reason: str, ref_price=None):
    try:
        trade_id = trade["id"]; symbol = trade.get("symbol","?")
        side = "sell"  # closing
        if SIMULATION_MODE:
            fill_px, latency_ms = _simulate_close(ref_price or 0, side)
        else:
            t0 = time.time()
            close_trade(trade)                       # real Tradier
            latency_ms = int((time.time()-t0)*1000)
            fill_px = trade.get("close_price") or ref_price

        reward = compute_shaped_reward(trade)
        next_state = build_meta_state_for_exit(trade)

        trade.update({
            "exit_reason": reason,
            "shaped_reward": reward,
            "meta_next_state": next_state,
            "close_price": fill_px,
            "latency_ms": latency_ms,
        })
        trade_tracker.mark_trade_closed(trade_id)
        log_trade_exit(trade)

        # log experience for PPO
        with open(META_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "state": trade.get("meta_state"),
                "action": trade.get("meta_action"),
                "reward": reward,
                "next_state": next_state,
                "done": True
            }) + "\n")

        notifier.send_message(
            f"🔴 Exit {symbol} ({trade_id})\n"
            f"{reason} | Reward {reward:.3f}"
        )
        bot_logger.info(f"[EXIT] {symbol} {trade_id} closed ({reason})")

    except Exception as exc:
        bot_logger.error(f"[Exit‑close] {exc}")
        bot_logger.debug(traceback.format_exc())