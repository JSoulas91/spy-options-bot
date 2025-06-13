"""
Signal‑generation and in‑position evaluation logic.

• `evaluate_trade_signal()` – produces a trade signal for entry.py
• `evaluate_trade()`        – decides hold / exit for open positions
• Uses meta‑agent overrides + market‑regime filters
"""

from __future__ import annotations
import traceback
from datetime import datetime, time
from typing import Dict, Any

from strategy.helpers import is_day_trade, is_swing_trade
from config import (
    CONFIDENCE_THRESHOLD, STOP_LOSS_ATR_MULTIPLIER,
    ENABLE_VIX_THROTTLING, VIX_MAX_THRESHOLD, VIX_MODERATE_THRESHOLD,
    CONFIDENCE_STEP_UP, MIN_META_CONFIDENCE
)
from utils.logger import bot_logger as logger
from utils.telegram_utils import send_telegram_message
from strategy.event_filter import is_high_risk_event_active
from utils.vix_utils import get_current_vix
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.options_fetcher import get_option_metrics
from meta.meta_agent import MetaAgent
from meta.reward_shaper import (
    compute_shaped_reward, compute_sharpe_style_reward, log_reward_trend
)
from meta.meta_state import normalize_meta_state

# ─────────────────────────────────────────────────────────
meta_agent = MetaAgent()

# ─────────────────────────────────────────────────────────
# ----------   1.  Trade‑Signal (entry side)   ------------
# ─────────────────────────────────────────────────────────
def evaluate_trade_signal(market_snap: Dict) -> Dict[str, Any]:
    try:
        if is_high_risk_event_active():
            logger.info("[Strategy] High-risk event active, skipping trade signal.")
            return _no_trade()

        vix = get_current_vix()
        if ENABLE_VIX_THROTTLING and vix > VIX_MAX_THRESHOLD:
            logger.info(f"[Strategy] VIX {vix} above max threshold, skipping signal.")
            return _no_trade()

        tf_1m = market_snap.get("tf_1m")
        tf_5m = market_snap.get("tf_5m")
        tf_15m = market_snap.get("tf_15m")
        tf_1h = market_snap.get("tf_1h")
        tf_1d = market_snap.get("tf_1d")
        long_term_data = fetch_long_term_features("SPY")

        atr = tf_1d["atr"].iloc[-1] if "atr" in tf_1d.columns else 2.0
        confidence = market_snap.get("confidence", 0.0)

        regime_features = {
            "confidence": confidence,
            "vix": vix,
            "hour": datetime.now().hour,
            "is_swing": float(is_swing_trade()),
            "atr": atr,
        }

        meta_state = normalize_meta_state(regime_features)
        meta_decision = meta_agent.should_enter(meta_state)

        if not meta_decision["should_enter"]:
            logger.info("[Strategy] Meta-agent vetoed entry.")
            return _no_trade()

        # Meta-agent approved, boost confidence
        confidence += CONFIDENCE_STEP_UP
        if confidence < MIN_META_CONFIDENCE:
            logger.info(f"[Strategy] Confidence {confidence:.2f} below threshold.")
            return _no_trade()

        trade_type = 1 if is_swing_trade() else 0
        trade_setup = {
            "symbol": "SPY",
            "stop_loss_pct": atr * STOP_LOSS_ATR_MULTIPLIER / 100,
        }

        logger.info(f"[Strategy] Entry signal passed with confidence {confidence:.2f}")
        return {
            "should_trade": True,
            "confidence": confidence,
            "trade_type": trade_type,
            "trade_setup": trade_setup,
            "tf_1m": tf_1m,
            "tf_5m": tf_5m,
            "tf_15m": tf_15m,
            "tf_1h": tf_1h,
            "tf_1d": tf_1d,
            "long_term_data": long_term_data,
        }

    except Exception as e:
        logger.error(f"[Strategy] evaluate_trade_signal error: {e}")
        logger.debug(traceback.format_exc())
        return _no_trade()

# ─────────────────────────────────────────────────────────
# ----------   2.  Exit Logic (in‑position eval)   --------
# ─────────────────────────────────────────────────────────
def evaluate_trade(trade: dict, spy_price: float, current_time=None) -> Dict[str, Any]:
    try:
        if is_high_risk_event_active():
            logger.info("[Strategy] High-risk event, recommending exit.")
            return {"exit": True, "reason": "event_risk"}

        entry_price = trade.get("entry_price", 0)
        atr = trade.get("atr", 2.0)
        stop_loss_pct = atr * STOP_LOSS_ATR_MULTIPLIER / 100
        stop_loss_price = entry_price * (1 - stop_loss_pct)

        if spy_price < stop_loss_price:
            logger.info(f"[Strategy] Price below stop-loss: {spy_price:.2f} < {stop_loss_price:.2f}")
            return {"exit": True, "reason": "stop_loss"}

        # Meta-agent override
        meta_state = trade.get("meta_state")
        if meta_state:
            decision = meta_agent.should_exit(meta_state)
            if decision["should_exit"]:
                logger.info("[Strategy] Meta-agent triggered exit.")
                return {"exit": True, "reason": "meta_exit"}

        return {"exit": False}

    except Exception as e:
        logger.error(f"[Strategy] evaluate_trade error: {e}")
        logger.debug(traceback.format_exc())
        return {"exit": False}

# ─────────────────────────────────────────────────────────
# ----------   3.  Post‑exit Reward Update   --------------
# ─────────────────────────────────────────────────────────
def post_exit_update(trade: dict, exit_price: float) -> None:
    try:
        shaped_reward = compute_shaped_reward(trade, exit_price)
        sharpe_score = compute_sharpe_style_reward(trade, exit_price)
        meta_agent.record_experience(trade, reward=shaped_reward)
        log_reward_trend(trade, shaped_reward, sharpe_score)
        logger.info(f"[Strategy] Reward logged. R: {shaped_reward:.3f}, Sharpe: {sharpe_score:.2f}")
    except Exception as e:
        logger.error(f"[Strategy] post_exit_update error: {e}")
        logger.debug(traceback.format_exc())

# ─────────────────────────────────────────────────────────
def _no_trade() -> Dict[str, Any]:
    return {
        "should_trade": False,
        "confidence": 0.0,
        "trade_type": 0,
        "trade_setup": {},
        "tf_1m": None,
        "tf_5m": None,
        "tf_15m": None,
        "tf_1h": None,
        "tf_1d": None,
        "long_term_data": {},
    }