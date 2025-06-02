# trade_manager.py

import time
import traceback
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
from meta.meta_agent import should_retry_trade  # Meta-agent veto logic

def should_hold_swing_trade(confidence: float, vix: float, is_monday_risk: bool) -> bool:
    """
    Determines if it's safe to hold a swing trade over the weekend.
    """
    if confidence < 0.7:
        return False
    if vix > 20:
        return False
    if is_monday_risk:
        return False
    return True

def execute_trade_with_retries(trade_function, contract, tracker: TradeTracker):
    retries = 0
    contract["retries_used"] = 0

    while retries < MAX_RETRIES:
        try:
            # Prevent trading expired contracts
            if contract.get("dte", 1) <= 0:
                logger.warning("⛔ Contract is expiring today. Trade skipped.")
                send_telegram_message("⛔ Trade blocked: Contract is expiring today.")
                return False

            # Enforce PDT if enabled
            if ENFORCE_PDT_LIMITS and not tracker.can_place_trade():
                logger.warning("🚫 PDT rule triggered. Trade skipped.")
                send_telegram_message("🚫 PDT rule triggered. Trade skipped.")
                return False

            # Meta-agent veto on retry
            if retries > 0:
                should_retry = should_retry_trade(contract)
                if not should_retry:
                    logger.warning("🧠 Meta-agent vetoed further retries.")
                    send_telegram_message("🧠 Meta-agent vetoed retry for this trade.")
                    return False

            # Time the trade for latency awareness
            start_time = time.time()
            success = trade_function(contract)
            latency = round(time.time() - start_time, 2)
            logger.info(f"⏱️ Trade execution latency: {latency}s")

            if success:
                contract["retries_used"] = retries
                return True
            else:
                raise Exception("Trade function returned False")

        except Exception as e:
            logger.error(f"⚠️ Trade attempt {retries + 1} failed: {e}")
            logger.debug(traceback.format_exc())
            retries += 1
            contract["retries_used"] = retries

            if retries < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * (2 ** (retries - 1))  # exponential backoff
                logger.info(f"🔁 Retrying in {delay} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)

    logger.error("❌ All trade attempts failed.")
    send_telegram_message("❌ All trade attempts failed after retries.")
    return False

def evaluate_swing_hold(contract: dict, confidence: float) -> bool:
    """
    Determines if this trade should be held as a swing over the weekend.
    """
    vix = get_current_vix()
    is_monday_risk = has_monday_macro_event()
    decision = should_hold_swing_trade(confidence, vix, is_monday_risk)

    if decision:
        logger.info("📊 Swing hold conditions met.")
        send_telegram_message(f"📊 Holding swing trade over weekend. VIX: {vix}, Confidence: {confidence}")
    else:
        logger.info("❌ Not holding swing trade: Failed safety criteria.")
        send_telegram_message(f"❌ Swing hold blocked. VIX: {vix}, Confidence: {confidence}, Monday risk: {is_monday_risk}")

    return decision