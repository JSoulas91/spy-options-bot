# strategy/strategy.py
import time
import traceback
from datetime import datetime, time as dt_time

from helpers import is_day_trade, is_swing_trade
from config import (
    CONFIDENCE_THRESHOLD,
    STOP_LOSS_ATR_MULTIPLIER,
    TRAILING_STOP_PERCENT,
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

# ---------------------------------------------------------------------
# In‑module lightweight caches to avoid hammering Tradier
# ---------------------------------------------------------------------
_MTF_CACHE = {}          # {symbol: (data, timestamp)}
_MTF_TTL   = 30          # seconds

_OPT_CACHE = {}          # {symbol: (metrics, timestamp)}
_OPT_TTL   = 60          # seconds

def _cached_mtf(symbol):
    now = time.time()
    data, ts = _MTF_CACHE.get(symbol, (None, 0))
    if data and now - ts < _MTF_TTL:
        return data
    data = fetch_long_term_features(symbol)
    _MTF_CACHE[symbol] = (data, now)
    return data

def _cached_option_metrics(symbol):
    now = time.time()
    data, ts = _OPT_CACHE.get(symbol, (None, 0))
    if data and now - ts < _OPT_TTL:
        return data
    data = get_option_metrics(symbol)
    _OPT_CACHE[symbol] = (data, now)
    return data

# ---------------------------------------------------------------------
# Meta‑agent initialization
# ---------------------------------------------------------------------
meta_agent = MetaAgent()
meta_agent.load_model()
RETURN_META_FEEDBACK = False


def is_market_closing_soon(ts_str: str) -> bool:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").time() >= dt_time(15, 55)
    except Exception:
        return False


def merge_indicators(primary: dict, fallback: dict) -> dict:
    out = fallback.copy()
    out.update({k: v for k, v in primary.items() if v is not None})
    return out


def extract_indicators(ind: dict) -> dict:
    keys = [
        "rsi", "atr", "vwap", "bb_upper", "bb_lower",
        "ema_50", "ema_200", "macd", "macd_signal",
        "support", "resistance",
        "implied_volatility", "delta", "gamma", "theta", "vega",
    ]
    return {k: ind.get(k) for k in keys}


# ---------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------
def evaluate_trade(position: dict, market_data: dict):
    """
    Decide 'hold' or 'exit' for an open position.
    Uses cached MTF + option data to stay well under 60 API calls/min.
    """
    try:
        symbol       = position.get("symbol", "SPY")
        entry_price  = position.get("entry_price")
        price        = market_data.get("price")
        timestamp    = market_data.get("timestamp", "")
        indicators   = market_data.get("indicators", {})
        confidence   = market_data.get("confidence_score", 0)

        # Fallback safety
        if entry_price is None or price is None:
            raise ValueError("Position missing 'entry_price' or market_data missing 'price'.")

        # =============== 1) Multi‑time‑frame indicators =================
        mtf_data = (
            market_data.get("long_term_data")              # upstream may supply
            or _cached_mtf(symbol)                         # cached fetch
        )
        merged_ind = merge_indicators(indicators, mtf_data.get("merged", {}))

        # =============== 2) Option Greeks / IV ==========================
        option_metrics = (
            market_data.get("option_metrics")              # upstream may supply
            or _cached_option_metrics(symbol)              # cached fetch
        )
        if option_metrics:
            merged_ind.update(option_metrics)

        # ---------------- Risk event block ----------------
        if is_high_risk_event_active():
            logger.warning("🚨 Economic event detected — exiting.")
            send_telegram_message("🚨 *Economic Event*\nAuto‑exit for risk control.")
            return "exit"

        # ---------------- Volatility inputs ---------------
        vix        = get_current_vix()
        trade_type = "day" if is_day_trade(position) else "swing"
        atr        = merged_ind.get("atr", 0)

        # ---------------- Meta‑agent state ----------------
        meta_state = normalize_meta_state({
            "confidence": confidence,
            "vix":        vix,
            "hour":       datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour if timestamp else 12,
            "is_swing":   1 if trade_type == "swing" else 0,
            "atr":        atr,
        })
        meta_action = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(meta_action)

        # --------------- Reward shaping (for log only) ----
        shaped = compute_shaped_reward(position, merged_ind, market_data)
        sharpe = compute_sharpe_style_reward(entry_price, price, atr)
        log_reward_trend({
            "meta_state":   meta_state,
            "meta_action":  meta_action,
            "meta_params":  meta_params,
            "shaped_reward":shaped,
            "sharpe_reward":sharpe,
        })

        # --------------- Meta‑agent overrides -------------
        if meta_action in (0, 2):
            return "exit"

        # --------------- Dynamic confidence threshold -----
        thresh = meta_params.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        if ENABLE_VIX_THROTTLING and vix is not None:
            if vix >= VIX_MAX_THRESHOLD:
                return "exit"
            if vix >= VIX_MODERATE_THRESHOLD:
                thresh += CONFIDENCE_STEP_UP
        if confidence < thresh:
            return "exit"

        # --------------- Additional exits -----------------
        extracted = extract_indicators(merged_ind)
        iv = extracted.get("implied_volatility")

        # Volatility‑based risk
        if vix and iv and vix > 22 and iv > 0.5:
            return "exit"

        # Time / ATR exits
        if is_day_trade(position):
            if is_market_closing_soon(timestamp):
                return "exit"
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                if price <= price * 0.90:
                    return "exit"
            if atr and price <= entry_price - atr * STOP_LOSS_ATR_MULTIPLIER:
                return "exit"
        else:  # swing logic
            if atr and price <= entry_price - atr * STOP_LOSS_ATR_MULTIPLIER * 1.5:
                return "exit"
            if iv and iv > 0.6:
                logger.warning("⚠️  High IV on swing — consider exit.")
        # --------------------------------------------------
        return "hold"

    except Exception as e:
        logger.error(f"[Strategy] evaluate_trade error: {e}\n{traceback.format_exc()}")
        return "hold"