# trade_manager.py
import time
import traceback
from typing import Optional, Dict, Any

from config import (
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    ENFORCE_PDT_LIMITS,
)
from utils.logger import bot_logger as logger
from utils.telegram import send_telegram_message
from utils.trade_tracker import TradeTracker
from utils.vix_utils import get_current_vix
from utils.economic_calendar import has_monday_macro_event
from meta.meta_agent import should_retry_trade

# Low‑level broker call (direct import from data layer)
from data.tradier_api import place_option_order

# ─────────────────────────────────────────────
# ░▒▓  Helper logic  ▓▒░
# ─────────────────────────────────────────────
def should_hold_swing_trade(confidence: float, vix: float, is_monday_risk: bool) -> bool:
    return all([
        confidence >= 0.7,
        vix <= 20,
        not is_monday_risk,
    ])

# ─────────────────────────────────────────────
# ░▒▓  Order execution with retries  ▓▒░
# ─────────────────────────────────────────────
def _submit_order(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts fields from contract dict and submits the trade.
    Required keys in contract: symbol, qty, side, order_type, duration
    """
    return place_option_order(
        option_symbol=contract["symbol"],
        quantity     =contract["qty"],
        side         =contract["side"],                 # e.g. 'buy_to_open'
        order_type   =contract.get("order_type", "market"),
        duration     =contract.get("duration", "day"),
    )

def execute_trade_with_retries(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt to place a trade up to MAX_RETRIES with exponential back‑off."""
    tracker = TradeTracker()
    retries = 0
    contract["retries_used"] = 0

    while retries < MAX_RETRIES:
        try:
            # ───── Preconditions ──────────────────────────
            if contract.get("dte", 1) <= 0:
                msg = "⛔ Contract is expiring today. Trade skipped."
                logger.warning(msg)
                send_telegram_message(msg)
                return None

            if ENFORCE_PDT_LIMITS and not tracker.can_place_trade():
                msg = "🚫 PDT rule triggered. Trade skipped."
                logger.warning(msg)
                send_telegram_message(msg)
                return None

            if retries > 0 and not should_retry_trade(contract):
                msg = "🧠 Meta‑agent vetoed further retries."
                logger.warning(msg)
                send_telegram_message(msg)
                return None

            # ───── Submit order ───────────────────────────
            order = _submit_order(contract)

            # Expect Tradier: {"id": "...", "status": "ok"/"filled"/"pending" ...}
            if order and order.get("status", "").lower() in {"ok", "filled", "open", "pending"}:
                contract["retries_used"] = retries
                tracker.record_trade(order)          # ensure PDT tracker stays accurate
                logger.info(f"✅ Order placed. ID: {order.get('id')}, Status: {order.get('status')}")
                return order
            else:
                raise Exception(f"Order rejected or malformed: {order}")

        except Exception as exc:
            logger.error(f"⚠️ Trade attempt {retries + 1} failed: {exc}")
            logger.debug(traceback.format_exc())
            retries += 1
            contract["retries_used"] = retries

            if retries < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * (2 ** (retries - 1))
                logger.info(f"🔁 Retrying in {delay} s … (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)

    logger.error("❌ All trade attempts failed.")
    send_telegram_message("❌ All trade attempts failed after retries.")
    return None

# ─────────────────────────────────────────────
# ░▒▓  Swing‑hold decision helper  ▓▒░
# ─────────────────────────────────────────────
def evaluate_swing_hold(contract: Dict[str, Any], confidence: float) -> bool:
    vix = get_current_vix()
    is_monday_risk = has_monday_macro_event()
    decision = should_hold_swing_trade(confidence, vix, is_monday_risk)

    if decision:
        msg = f"📊 Holding swing trade over weekend. VIX: {vix}, Confidence: {confidence}"
        logger.info(msg)
        send_telegram_message(msg)
    else:
        msg = f"❌ Swing hold blocked. VIX: {vix}, Confidence: {confidence}, Monday risk: {is_monday_risk}"
        logger.info(msg)
        send_telegram_message(msg)

    return decision