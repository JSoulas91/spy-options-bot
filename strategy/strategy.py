import traceback
from datetime import datetime, time
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
from meta.reward_shaper import compute_shaped_reward, compute_sharpe_style_reward, log_reward_trend
from meta.meta_state import normalize_meta_state
from data.multi_timeframe_fetcher import fetch_long_term_features
from data.options_fetcher import get_option_metrics

meta_agent = MetaAgent()
meta_agent.load_model()

RETURN_META_FEEDBACK = False

def is_market_closing_soon(timestamp_str):
    try:
        current_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").time()
        return current_time >= time(15, 55)
    except Exception:
        return False

def merge_indicators(primary, fallback):
    # Merge dicts with preference for primary values if present
    merged = fallback.copy()
    merged.update({k: v for k, v in primary.items() if v is not None})
    return merged

def extract_indicators(indicators):
    # Extract known indicators safely, fallback None if missing
    keys = [
        "rsi", "atr", "vwap", "bb_upper", "bb_lower",
        "ema_50", "ema_200", "macd", "macd_signal",
        "support", "resistance",
        "implied_volatility", "delta", "gamma", "theta", "vega"
    ]
    return {key: indicators.get(key) for key in keys}

def evaluate_trade(position, market_data):
    try:
        action = "hold"
        symbol = position.get("symbol", "SPY")
        entry_price = position.get("entry_price")
        price = market_data.get("price")
        timestamp = market_data.get("timestamp", "")
        indicators = market_data.get("indicators", {})
        confidence = market_data.get("confidence_score", 0)

        if entry_price is None or price is None:
            raise ValueError("Missing 'entry_price' or 'price'.")

        # Fetch multi-timeframe indicators (long-term & intraday)
        try:
            mtf_indicators = fetch_long_term_features(symbol)
            merged_indicators = merge_indicators(indicators, mtf_indicators.get("merged", {}))
            logger.info(f"[MTF] Merged multi-timeframe indicators for {symbol}")
        except Exception as e:
            logger.error(f"[MTF Error] Failed fetching multi-timeframe indicators for {symbol}: {e}")
            merged_indicators = indicators

        # Fetch option Greeks and implied volatility metrics
        try:
            option_metrics = get_option_metrics(symbol)
            if option_metrics:
                merged_indicators.update(option_metrics)
                logger.info(f"[Greeks] Added option metrics for {symbol}")
        except Exception as e:
            logger.error(f"[Greeks Error] Failed to fetch option metrics: {e}")

        # Risk event check: exit immediately if live economic event detected
        if is_high_risk_event_active():
            logger.warning("🚨 Live economic event detected — exiting position to reduce risk.")
            send_telegram_message("🚨 *Live Economic Event Detected*\nAuto-exiting position to reduce risk exposure.")
            return ("exit", None) if RETURN_META_FEEDBACK else "exit"

        # Get VIX for volatility throttling
        vix = get_current_vix()
        trade_type = "day" if is_day_trade(position) else "swing"
        hour = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour if timestamp else 12
        atr = merged_indicators.get("atr", 0)

        # Prepare meta-agent input state & normalize
        meta_input = {
            "confidence": confidence,
            "vix": vix,
            "hour": hour,
            "is_swing": 1 if trade_type == "swing" else 0,
            "atr": atr
        }
        meta_state = normalize_meta_state(meta_input)

        # Meta-agent decision
        meta_action = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(meta_action)

        # Compute shaped rewards for training feedback
        shaped_reward = compute_shaped_reward(position, merged_indicators, market_data)
        sharpe_reward = compute_sharpe_style_reward(entry_price, price, atr)
        meta_feedback = {
            "meta_state": meta_state,
            "meta_action": meta_action,
            "meta_params": meta_params,
            "shaped_reward": shaped_reward,
            "sharpe_reward": sharpe_reward
        }
        log_reward_trend(meta_feedback)

        logger.info(f"🧠 Meta-State: {meta_state} | Action: {meta_action} | Params: {meta_params}")
        logger.info(f"🎯 Rewards — Shaped: {shaped_reward:.4f} | Sharpe-style: {sharpe_reward:.4f}")

        # Meta-agent forced skip or exit
        if meta_action == 0:
            logger.info("🧠 Meta-agent recommends skipping this trade.")
            return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"
        if meta_action == 2:
            logger.info("🧠 Meta-agent enforces exit — overriding strategy.")
            return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

        # Adaptive confidence threshold with VIX throttling
        adaptive_threshold = meta_params.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        if ENABLE_VIX_THROTTLING or ENABLE_ADAPTIVE_CONFIDENCE:
            try:
                if vix is not None:
                    if vix >= VIX_MAX_THRESHOLD:
                        logger.warning("❌ VIX above max threshold — skipping trade.")
                        return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"
                    elif vix >= VIX_MODERATE_THRESHOLD:
                        adaptive_threshold += CONFIDENCE_STEP_UP
                        logger.info(f"⚠️ Elevated VIX — raising confidence threshold to {adaptive_threshold:.2f}")
            except Exception as e:
                logger.error(f"[VIX Error] Threshold adjustment failed: {e}")

        if confidence < adaptive_threshold:
            logger.info(f"⚠️ Confidence {confidence:.2f} below adaptive threshold {adaptive_threshold:.2f} — exiting.")
            return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

        extracted = extract_indicators(merged_indicators)
        iv = extracted.get("implied_volatility")

        # Elevated volatility risks
        if vix and iv and vix > 22 and iv > 0.5:
            logger.warning(f"🚨 Elevated VIX ({vix:.2f}) + High IV ({iv:.2f}) — risk too high.")
            return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

        # Day trade specific exit logic
        if is_day_trade(position):
            if is_market_closing_soon(timestamp):
                logger.info("⏰ Market is closing soon — exiting day trade.")
                return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

            # Trailing stop-loss check (price rise then drop)
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                # Example: trailing stop triggers if price falls 10% from recent high (adjust logic as needed)
                # Here simplified, replace with your trailing stop logic as you prefer
                trailing_stop_price = price * (1 - 0.10)
                if price <= trailing_stop_price:
                    logger.info("📉 Trailing stop-loss triggered after gain.")
                    return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

            # ATR stop-loss for day trades
            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER):
                logger.info("🛑 ATR stop-loss hit (day trade).")
                return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

        # Swing trade specific exit logic
        elif is_swing_trade(position):
            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER * 1.5):
                logger.info("🛑 ATR stop-loss hit (swing trade).")
                return ("exit", meta_feedback) if RETURN_META_FEEDBACK else "exit"

            if iv and iv > 0.6:
                logger.warning(f"⚠️ High implied volatility ({iv:.2f}) on swing trade — consider exit.")
                # You could optionally enforce exit here or just warn

        return ("hold", meta_feedback) if RETURN_META_FEEDBACK else "hold"

    except Exception as e:
        logger.error(f"[Evaluate Trade Error] {e}\n{traceback.format_exc()}")
        return ("hold", None) if RETURN_META_FEEDBACK else "hold"