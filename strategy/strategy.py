import traceback
from datetime import datetime, time
from helpers import is_day_trade, is_swing_trade
from config import (
    CONFIDENCE_THRESHOLD,
    STOP_LOSS_ATR_MULTIPLIER,
    ENABLE_VIX_THROTTLING,
    ENABLE_ADAPTIVE_CONFIDENCE,
    VIX_MAX_THRESHOLD,
    VIX_MODERATE_THRESHOLD,
    CONFIDENCE_STEP_UP,
)
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from event_filter import is_high_risk_event_active
from utils.vix_utils import get_current_vix
from meta.meta_agent import MetaAgent
from meta.reward_shaper import (
    compute_shaped_reward,
    compute_sharpe_style_reward,
    log_reward_trend,
)
from meta.meta_state import normalize_meta_state
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.options_fetcher import get_option_metrics

meta_agent = MetaAgent()
meta_agent.load_model()

RETURN_META_FEEDBACK = False

def is_market_closing_soon(ts: str) -> bool:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").time() >= time(15, 55)
    except Exception:
        return False

def merge_indicators(primary: dict, fallback: dict):
    merged = fallback.copy()
    merged.update({k: v for k, v in primary.items() if v is not None})
    return merged

KEYS = [
    "rsi","atr","vwap","bb_upper","bb_lower",
    "ema_50","ema_200","macd","macd_signal",
    "support","resistance","implied_volatility",
    "delta","gamma","theta","vega"
]
def extract(ind): return {k: ind.get(k) for k in KEYS}

def evaluate_trade(position: dict, market_data: dict):
    try:
        symbol      = position.get("symbol", "SPY")
        entry_price = position.get("entry_price")
        price       = market_data.get("price")
        ts          = market_data.get("timestamp", "")
        indicators  = market_data.get("indicators", {})
        confidence  = market_data.get("confidence_score", 0)

        if entry_price is None or price is None:
            raise ValueError("Missing entry_price or price.")

        # ── Merge indicators
        try:
            mtf = fetch_long_term_features(symbol)
            ind = merge_indicators(indicators, mtf.get("merged", {}))
        except Exception as e:
            logger.error(f"[MTF] {e}"); ind = indicators
        try:
            opts = get_option_metrics(symbol)
            if opts: ind.update(opts)
        except Exception as e:
            logger.error(f"[Greeks] {e}")

        # ── Immediate exit if risk event
        if is_high_risk_event_active():
            send_telegram_message("🚨 Econ event — exit")
            return "exit"

        vix        = get_current_vix()
        trade_type = "day" if is_day_trade(position) else "swing"
        hour       = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").hour if ts else 12
        atr        = ind.get("atr", 0)

        meta_in  = {"confidence":confidence,"vix":vix,"hour":hour,
                    "is_swing":1 if trade_type=="swing" else 0,"atr":atr}
        meta_st  = normalize_meta_state(meta_in)
        meta_act = meta_agent.select_action(meta_st)
        meta_par = meta_agent.interpret_action(meta_act)

        shaped = compute_shaped_reward(position, ind, market_data)
        sharpe = compute_sharpe_style_reward(entry_price, price, atr)
        log_reward_trend({"meta_state":meta_st,"meta_action":meta_act,
                          "shaped_reward":shaped,"sharpe_reward":sharpe})

        # meta overrides
        if meta_act in (0,2):
            return "exit"

        # confidence / VIX gate
        thr = meta_par.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        if ENABLE_VIX_THROTTLING and vix:
            if vix >= VIX_MAX_THRESHOLD: return "exit"
            if vix >= VIX_MODERATE_THRESHOLD: thr += CONFIDENCE_STEP_UP
        if confidence < thr: return "exit"

        iv = ind.get("implied_volatility")

        if vix and iv and vix > 22 and iv > 0.5:
            return "exit"

        # ── DAY‑TRADE EXIT LOGIC (ATR high‑water trailing)
        if is_day_trade(position):
            if is_market_closing_soon(ts): return "exit"

            high_mark = position.get("high_since_entry", entry_price)
            if price > high_mark:
                position["high_since_entry"] = price
                high_mark = price

            trail = atr * 1.2 if atr else price * 0.015
            if price <= high_mark - trail:
                logger.info("📉 ATR trailing stop triggered.")
                return "exit"

            if atr and price <= entry_price - atr * STOP_LOSS_ATR_MULTIPLIER:
                return "exit"

        # ── SWING
        elif is_swing_trade(position):
            if atr and price <= entry_price - atr * STOP_LOSS_ATR_MULTIPLIER * 1.5:
                return "exit"
            if iv and iv > 0.6:
                logger.warning("High IV swing – consider exit.")

        return "hold"
    except Exception as e:
        logger.error(f"[Strategy] {e}\n{traceback.format_exc()}")
        return "hold"