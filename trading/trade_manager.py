import time
import traceback
from typing import Optional, Dict, Any

from config import MAX_RETRIES, RETRY_DELAY_SECONDS, ENFORCE_PDT_LIMITS
from utils.logger import bot_logger as logger
from utils.telegram_utils import send_telegram_message
from utils.trade_tracker import TradeTracker
from utils.vix_utils import get_current_vix
from utils.economic_calendar import week_has_fomc_or_cpi
from meta.meta_agent import should_retry_trade
from data.tradier_api import place_option_order

# ─── Health‑check integration ────────────────────────────────────────────────
from monitor.health_check import update_status

# ─── Constants ───────────────────────────────────────────────────────────────
VALID_ORDER_STATUSES = {"ok", "filled", "open", "pending"}

# ─── Cached macro/VIX values ─────────────────────────────────────────────────
_cached_vix = None
_cached_macro = None

# ─── Helpers ─────────────────────────────────────────────────────────────────
def should_hold_swing_trade(confidence: float, vix: float, is_monday_risk: bool) -> bool:
    return all([confidence >= 0.7, vix <= 20, not is_monday_risk])

# --------------------------------------------------------------------------- #
def _submit_order(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return place_option_order(
        option_symbol=contract["symbol"],
        quantity     =contract["qty"],
        side         =contract["side"],
        order_type   =contract.get("order_type", "market"),
        duration     =contract.get("duration", "day"),
    )

def execute_trade_with_retries(contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tracker = TradeTracker()
    retries = 0
    contract["retries_used"] = 0

    while retries < MAX_RETRIES:
        try:
            update_status("last_trade_attempt")  # ✅

            # Preconditions
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

            # Submit order
            order = _submit_order(contract)

            if order and order.get("status", "").lower() in VALID_ORDER_STATUSES:
                contract["retries_used"] = retries
                tracker.record_trade(order)
                update_status("last_trade")  # ✅ confirmed trade
                order_id = order.get("id", "UNKNOWN")
                logger.info(f"✅ Order placed. ID: {order_id}, Status: {order.get('status')}")
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

# --------------------------------------------------------------------------- #
def evaluate_swing_hold(contract: Dict[str, Any], confidence: float) -> bool:
    global _cached_vix, _cached_macro

    if _cached_vix is None:
        _cached_vix = get_current_vix()

    if _cached_macro is None:
        _cached_macro = week_has_fomc_or_cpi()

    decision = should_hold_swing_trade(confidence, _cached_vix, _cached_macro)

    if decision:
        msg = f"📊 Holding swing trade over weekend. VIX: {_cached_vix}, Confidence: {confidence}"
    else:
        msg = f"❌ Swing hold blocked. VIX: {_cached_vix}, Confidence: {confidence}, Monday risk: {_cached_macro}"

    logger.info(msg)
    send_telegram_message(msg)

    return decision

# ──────────────────────────────────────────────────────────────────────────────
def close_trade(trade: dict) -> None:
    """
    Closes an open trade.
    Here, implement actual closing logic (e.g. market sell orders).
    This is a placeholder for integration with Tradier or Alpaca order APIs.
    """
    try:
        # Example placeholder logic:
        # symbol = trade.get("symbol")
        # qty = trade.get("qty")
        # place a market order to close position
        logger.info(f"[TradeManager] Closing trade {trade.get('id')} for {trade.get('symbol')}")
        # TODO: add actual close order logic here
        # e.g. call place_option_order or Alpaca API with side='sell' and qty
        time.sleep(0.5)  # simulate latency
    except Exception as exc:
        logger.error(f"[TradeManager] Error closing trade {trade.get('id')}: {exc}")
        raise