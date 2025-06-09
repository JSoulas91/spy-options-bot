# strategy/strategy.py
"""
In‑position evaluation logic.

• Merges multi‑time‑frame indicators + option Greeks
• Uses meta‑agent override + market‑regime filter
• Confidence/VIX‑aware gates + multi‑action scaling
• Returns "hold" | "exit"
"""

import traceback
from datetime import datetime, time
from typing   import Dict, Any

from strategy.helpers import is_day_trade, is_swing_trade, get_min_meta_confidence
from config  import (
    CONFIDENCE_THRESHOLD, STOP_LOSS_ATR_MULTIPLIER,
    ENABLE_VIX_THROTTLING, VIX_MAX_THRESHOLD, VIX_MODERATE_THRESHOLD,
    CONFIDENCE_STEP_UP
)
from utils.logger          import bot_logger as logger
from utils.telegram_utils  import send_telegram_message
from strategy.event_filter          import is_high_risk_event_active
from utils.vix_utils       import get_current_vix
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.options_fetcher  import get_option_metrics

from meta.meta_agent       import MetaAgent
from meta.reward_shaper    import (
    compute_shaped_reward, compute_sharpe_style_reward, log_reward_trend
)
from meta.meta_state       import normalize_meta_state

# ─────────────────────────────────────────────────────────
meta_agent = MetaAgent()
meta_agent.load_model()

# ─────────────────────────────────────────────────────────
IND_KEYS = [
    "rsi","atr","vwap","bb_upper","bb_lower","ema_50","ema_200",
    "macd","macd_signal","support","resistance",
    "implied_volatility","delta","gamma","theta","vega"
]
def _extract(ind: Dict[str, Any]):
    return {k: ind.get(k) for k in IND_KEYS}

def _is_close_to_close(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").time() >= time(15, 55)
    except Exception:
        return False

def _merge(primary: Dict, fallback: Dict) -> Dict:
    d = fallback.copy()
    d.update({k: v for k, v in primary.items() if v is not None})
    return d

def classify_regime(one_day: dict, vix_val: float) -> str:
    price  = one_day.get("price", 0)
    ema200 = one_day.get("ema_200", price)
    if price > ema200 and vix_val < 18:
        return "bull"
    if price < ema200 and vix_val > 25:
        return "bear"
    return "vol_cluster"

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

        # ───── Market regime detection ──────────────────────────
        one_day = mtf["intraday"].get("1d_6mo", {}) if isinstance(mtf, dict) else {}
        vix_val = get_current_vix() or 20
        regime  = classify_regime(one_day, vix_val)

        # ───── Immediate risk exits ─────────────────────────────
        if is_high_risk_event_active():
            send_telegram_message("🚨 Econ event — exit all positions")
            return "exit"

        if ENABLE_VIX_THROTTLING and vix_val >= VIX_MAX_THRESHOLD:
            logger.info(f"⚠️ VIX {vix_val:.2f} above max — exiting.")
            return "exit"

        # ───── Meta‑agent evaluation ────────────────────────────
        trade_type = "day" if is_day_trade(position) else "swing"
        hour       = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").hour if ts else 12
        atr        = ind.get("atr", 0)

        meta_state = normalize_meta_state({
            "confidence": confidence,
            "vix": vix_val,
            "hour": hour,
            "is_swing": 1 if trade_type == "swing" else 0,
            "atr": atr,
        })
        meta_action = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(meta_action)

        shaped = compute_shaped_reward(position, ind, market_data)
        sharpe = compute_sharpe_style_reward(entry_px, price, atr)
        log_reward_trend({
            "meta_state": meta_state,
            "meta_action": meta_action,
            "shaped_reward": shaped,
            "sharpe_reward": sharpe,
        })

        if meta_action == 0:
            return "exit"

        agent_conf = meta_params.get("agent_confidence")
        if agent_conf is not None and agent_conf < get_min_meta_confidence(regime):
            return "exit"

        # ───── Confidence / VIX / Regime filter ─────────────────
        thr = meta_params.get("confidence_threshold", CONFIDENCE_THRESHOLD)

        if regime in {"bear", "vol_cluster"}:
            thr += 0.05

        if ENABLE_VIX_THROTTLING and vix_val >= VIX_MODERATE_THRESHOLD:
            thr += CONFIDENCE_STEP_UP

        if confidence < thr:
            return "exit"

        # ───── Additional risk checks ──────────────────────────
        iv = ind.get("implied_volatility")
        if vix_val > 22 and iv and iv > 0.5:
            return "exit"

        scale   = meta_params.get("scale_factor", 1.0)
        tighten = meta_params.get("tighten_exit", False)

        # ───── Day‑trade specific exits ────────────────────────
        if is_day_trade(position):
            if _is_close_to_close(ts):
                return "exit"

            high_mark = position.get("high_since_entry", entry_px)
            if price > high_mark:
                position["high_since_entry"] = price
                high_mark = price

            trail_base = atr * 1.2 if atr else price * 0.015
            trail = trail_base / scale
            if tighten:
                trail *= 0.8

            if price <= high_mark - trail:
                return "exit"

            stop_dist = atr * STOP_LOSS_ATR_MULTIPLIER / scale
            if tighten:
                stop_dist *= 0.8
            if atr and price <= entry_px - stop_dist:
                return "exit"

        # ───── Swing‑trade exits ───────────────────────────────
        elif is_swing_trade(position):
            stop_dist = atr * STOP_LOSS_ATR_MULTIPLIER * 1.5 / scale
            if tighten:
                stop_dist *= 0.8
            if atr and price <= entry_px - stop_dist:
                return "exit"
            if iv and iv > 0.6:
                logger.warning("High IV swing – consider exit.")

        return "hold"

    except Exception as e:
        logger.error(f"[Strategy] {e}\n{traceback.format_exc()}")
        return "hold"