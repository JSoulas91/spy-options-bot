"""
Signal‑generation and in‑position evaluation logic.

• `evaluate_trade_signal()` – produces a trade signal for entry.py
• `evaluate_trade()`        – decides hold / exit for open positions
• Uses meta‑agent overrides + market‑regime filters
"""

from __future__ import annotations
import traceback
from datetime import datetime
from typing import Dict, Any
import os
import joblib
import numpy as np
import pandas as pd

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

# Load calibrated classifier once at module load
CALIBRATED_MODEL_PATH = "models/xgb_calibrated.pkl"
calibrated_model = None
try:
    if os.path.exists(CALIBRATED_MODEL_PATH):
        calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)
        logger.info(f"[Strategy] Loaded calibrated classifier from {CALIBRATED_MODEL_PATH}")
    else:
        logger.warning(f"[Strategy] Calibrated model not found at {CALIBRATED_MODEL_PATH}")
except Exception as e:
    logger.error(f"[Strategy] Failed to load calibrated model: {e}")
    calibrated_model = None

def predict_trade_outcome(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict trade outcome probabilities and direction using the calibrated classifier.

    Args:
        features: Dict of features matching the classifier input schema.

    Returns:
        Dict with:
          - trade_success_prob (float)
          - predicted_direction (int)  # e.g., 1=long, 0=neutral, -1=short
          - class_probabilities (list of floats)
          - entropy (float)
    """
    if calibrated_model is None:
        # Model not loaded, return safe defaults
        logger.warning("[Strategy] Calibrated model unavailable, skipping prediction.")
        return {
            "trade_success_prob": 0.0,
            "predicted_direction": 0,
            "class_probabilities": [0.0, 0.0],
            "entropy": 0.0,
        }

    try:
        # Convert features dict to dataframe with single row
        input_df = pd.DataFrame([features])
        proba = calibrated_model.predict_proba(input_df)[0]
        pred_class = np.argmax(proba)
        entropy = -np.sum(proba * np.log(proba + 1e-12))  # add small epsilon to avoid log(0)

        # Assuming class 1 = success (long), class 0 = failure (neutral/short)
        trade_success_prob = proba[1]
        predicted_direction = 1 if pred_class == 1 else 0

        return {
            "trade_success_prob": float(trade_success_prob),
            "predicted_direction": int(predicted_direction),
            "class_probabilities": proba.tolist(),
            "entropy": float(entropy),
        }
    except Exception as e:
        logger.error(f"[Strategy] Prediction error: {e}")
        return {
            "trade_success_prob": 0.0,
            "predicted_direction": 0,
            "class_probabilities": [0.0, 0.0],
            "entropy": 0.0,
        }

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

        atr = tf_1d["atr"].iloc[-1] if tf_1d is not None and "atr" in tf_1d.columns else 2.0
        confidence = market_snap.get("confidence", 0.0)

        regime_features = {
            "confidence": confidence,
            "vix": vix,
            "hour": datetime.now().hour,
            "is_swing": float(is_swing_trade()),
            "atr": atr,
        }

        # Normalize meta state for meta-agent decision
        meta_state = normalize_meta_state(regime_features)
        meta_decision = meta_agent.should_enter(meta_state)

        if not meta_decision["should_enter"]:
            logger.info("[Strategy] Meta-agent vetoed entry.")
            return _no_trade()

        confidence += CONFIDENCE_STEP_UP
        if confidence < MIN_META_CONFIDENCE:
            logger.info(f"[Strategy] Confidence {confidence:.2f} below threshold.")
            return _no_trade()

        # Prepare features for classifier prediction
        # Use market_snap features plus regime features as needed
        # Assuming market_snap keys match classifier feature columns
        classifier_features = dict(market_snap)  # shallow copy
        # Add any additional features if needed, e.g.:
        classifier_features.update({
            "vix": vix,
            "hour": datetime.now().hour,
            "is_swing": float(is_swing_trade()),
            "atr": atr,
            "confidence": confidence,
        })

        clf_out = predict_trade_outcome(classifier_features)

        trade_type = 1 if is_swing_trade() else 0
        trade_setup = {
            "symbol": "SPY",
            "stop_loss_pct": atr * STOP_LOSS_ATR_MULTIPLIER / 100,
        }

        logger.info(f"[Strategy] Entry signal passed with confidence {confidence:.2f}, "
                    f"Classifier trade_success_prob={clf_out['trade_success_prob']:.3f}")

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
            # Classifier outputs included for downstream use
            "trade_success_prob": clf_out["trade_success_prob"],
            "predicted_direction": clf_out["predicted_direction"],
            "class_probabilities": clf_out["class_probabilities"],
            "entropy": clf_out["entropy"],
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
        # Classifier outputs default to zero
        "trade_success_prob": 0.0,
        "predicted_direction": 0,
        "class_probabilities": [0.0, 0.0],
        "entropy": 0.0,
    } 