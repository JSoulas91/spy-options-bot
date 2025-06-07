# strategy/strategy.py
"""
In‑position evaluation logic.

• Merges multi‑time‑frame indicators + option Greeks
• Uses meta‑agent override + market‑regime filter
• Confidence/VIX‑aware gates
• Returns "hold" | "exit"
"""

import traceback
from datetime import datetime, time
from typing import Dict, Any

from helpers            import is_day_trade, is_swing_trade
from config             import (
    CONFIDENCE_THRESHOLD, STOP_LOSS_ATR_MULTIPLIER,
    ENABLE_VIX_THROTTLING, VIX_MAX_THRESHOLD, VIX_MODERATE_THRESHOLD,
    CONFIDENCE_STEP_UP, MIN_META_CONFIDENCE       # ← NEW (.env)
)
from utils.logger       import bot_logger as logger
from utils.telegram_utils import send_telegram_message
from event_filter       import is_high_risk_event_active
from utils.vix_utils    import get_current_vix
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.options_fetcher   import get_option_metrics

from meta.meta_agent    import MetaAgent
from meta.reward_shaper import (
    compute_shaped_reward, compute_sharpe_style_reward, log_reward_trend
)
from meta.meta_state    import normalize_meta_state    # dict‑normaliser

# ─────────────────────────────────────────────────────────
meta_agent = MetaAgent()
meta_agent.load_model()

# ─────────────────────────────────────────────────────────
IND_KEYS = [
    "rsi","atr","vwap","bb_upper","bb_lower","ema_50","ema_200",
    "macd","macd_signal","support","resistance",
    "implied_volatility","delta","gamma","theta","vega"
]
def _extract(ind: Dict[str, Any]):       # keep only known keys
    return {k: ind.get(k) for k in IND_KEYS}

def _is_close_to_close(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").time() >= time(15, 55)
    except:                               # malformed timestamp
        return False

def _merge(primary: Dict, fallback: Dict) -> Dict:
    d = fallback.copy()
    d.update({k: v for k, v in primary.items() if v is not None})
    return d

# ─────────────────────────────────────────────────────────
def evaluate_trade(position: Dict, market_data: Dict) -> str:
    """
    Decide whether an open trade should be held or exited.
    """
    try:
        symbol      = position.get("symbol", "SPY")
        entry_px    = position.get("entry_price")
        price       = market_data.get("price")
        ts          = market_data.get("timestamp", "")
        indics      = _extract(market_data.get("indicators", {}))
        confidence  = market_data.get("confidence_score", 0)

        if entry_px is None or price is None:
            raise ValueError("entry_price or price missing.")

        # ───── Indicator merge (MTF + Greeks) ───────────────────
        try:
            mtf = fetch_long_term_features(symbol)
            ind = _merge(indics, mtf.get("merged", {}))
        except Exception as e:
            logger.error(f"[MTF] {e}")
            ind = indics

        try:
            greeks = get_option_metrics(symbol)
            if greeks:
                ind.update(greeks)
        except Exception as e:
            logger.error(f"[Greeks] {e}")

        # ───── Immediate risk exits ─────────────────────────────
        if is_high_risk_event_active():
            send_telegram_message("🚨 Economic event – closing positions.")
            return "exit"

        vix_val    = get_current_vix() or 20
        trade_type = "day" if is_day_trade(position) else "swing"
        hour       = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").hour if ts else 12
        atr        = ind.get("atr", 0)

        # ───── Meta‑agent override / parameters ─────────────────
        meta_state   = normalize_meta_state({
            "confidence": confidence,
            "vix":        vix_val,
            "hour":       hour,
            "is_swing":   1 if trade_type == "swing" else 0,
            "atr":        atr,
        })
        meta_action  = meta_agent.select_action(meta_state)
        meta_params  = meta_agent.interpret_action(meta_action)

        shaped = compute_shaped_reward(position, ind, market_data)
        sharpe = compute_sharpe_style_reward(entry_px, price, atr)
        log_reward_trend({
            "meta_state":    meta_state,
            "meta_action":   meta_action,
            "shaped_reward": shaped,
            "sharpe_reward": sharpe,
        })

        # meta_action semantics:
        #   0 -> force EXIT
        #   1 -> hold (normal)
        #   2 -> tighten exit rules (aggressive)
        if meta_action == 0:
            return "exit"

        # If the meta‑agent produced a confidence estimate, honour it
        agent_conf = meta_params.get("agent_confidence")
        if agent_conf is not None and agent_conf < MIN_META_CONFIDENCE:
            return "exit"

        # ───── Confidence / VIX filter ─────────────────────────
        thr = meta_params.get("confidence_threshold", CONFIDENCE_THRESHOLD)

        if ENABLE_VIX_THROTTLING:
            if vix_val >= VIX_MAX_THRESHOLD:
                return "exit"
            if vix_val >= VIX_MODERATE_THRESHOLD:
                thr += CONFIDENCE_STEP_UP

        if confidence < thr:
            return "exit"

        # ───── Additional risk checks ──────────────────────────
        iv = ind.get("implied_volatility")
        if vix_val > 22 and iv and iv > 0.5:
            return "exit"

        # ───── Day‑trade specific exits ────────────────────────
        if is_day_trade(position):
            if _is_close_to_close(ts):
                return "exit"

            high_mark = position.get("high_since_entry", entry_px)
            if price > high_mark:
                position["high_since_entry"] = price
                high_mark = price

            trail = atr * 1.2 if atr else price * 0.015
            if price <= high_mark - trail:
                return "exit"

            if atr and price <= entry_px - atr * STOP_LOSS_ATR_MULTIPLIER:
                return "exit"

        # ───── Swing‑trade exits ───────────────────────────────
        elif is_swing_trade(position):
            if atr and price <= entry_px - atr * STOP_LOSS_ATR_MULTIPLIER * 1.5:
                return "exit"
            if iv and iv > 0.6:
                logger.warning("High IV swing – consider exit.")

        return "hold"

    except Exception as e:
        logger.error(f"[Strategy] {e}\n{traceback.format_exc()}")
        return "hold"