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
from data.multi_timeframe_fetcher import get_multi_timeframe_data

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
    """Merge indicator dictionaries, preferring values from primary."""
    return {key: primary.get(key, fallback.get(key)) for key in set(primary) | set(fallback)}


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

        # Fetch and merge multi-timeframe indicators
        try:
            mtf_data = get_multi_timeframe_data(symbol)
            mtf_indicators = mtf_data.get("merged", {})
            indicators = merge_indicators(indicators, mtf_indicators)
            logger.info(f"[MTF] Merged indicators from multi-timeframe for {symbol}")
        except Exception as e:
            logger.error(f"[MTF Error] Failed to retrieve multi-timeframe data for {symbol}: {e}")

        # 🚨 Economic event risk off
        if is_high_risk_event_active():
            logger.warning("🚨 Live economic event detected — exiting to reduce risk.")
            send_telegram_message("🚨 *Live Economic Event Detected*\nAuto-exiting position to reduce risk exposure.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", None)

        # Build meta-agent state
        vix = get_current_vix()
        trade_type = "day" if is_day_trade(position) else "swing"
        hour = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour if timestamp else 12
        volatility = indicators.get("atr", 0)

        meta_state = [confidence, vix or 0, hour, 0 if trade_type == "day" else 1, volatility]
        meta_action = meta_agent.select_action(meta_state)
        meta_params = meta_agent.interpret_action(meta_action)
        meta_feedback = {
            "meta_state": meta_state,
            "meta_action": meta_action,
            "meta_params": meta_params
        }

        logger.info(f"🧠 Meta-State: {meta_state} | Meta-Action: {meta_action} | Meta-Params: {meta_params}")

        # Meta-agent override actions
        if meta_action == 0:
            logger.info("🧠 Meta-agent recommends skipping this trade.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)
        if meta_action == 2:
            logger.info("🧠 Meta-agent enforces exit — overriding strategy.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

        # Use adaptive threshold if specified
        adaptive_threshold = meta_params.get("confidence_threshold", CONFIDENCE_THRESHOLD)

        if ENABLE_VIX_THROTTLING or ENABLE_ADAPTIVE_CONFIDENCE:
            try:
                if vix is not None:
                    if vix >= VIX_MAX_THRESHOLD:
                        logger.warning("❌ VIX above max threshold — skipping trade.")
                        return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)
                    elif vix >= VIX_MODERATE_THRESHOLD:
                        adaptive_threshold += CONFIDENCE_STEP_UP
                        logger.info(f"⚠️ Elevated VIX — raising confidence threshold to {adaptive_threshold:.2f}")
            except Exception as e:
                logger.error(f"[VIX Error] Confidence threshold VIX adjustment failed: {e}")

        if confidence < adaptive_threshold:
            logger.info(f"⚠️ Confidence {confidence:.2f} below required {adaptive_threshold:.2f} — exiting.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

        # ⛓ Technical signal-based exit logic
        rsi = indicators.get('rsi')
        atr = indicators.get('atr')
        vwap = indicators.get('vwap')
        upper_band = indicators.get('bb_upper')
        lower_band = indicators.get('bb_lower')
        ema_50 = indicators.get('ema_50')
        ema_200 = indicators.get('ema_200')
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_hist = macd - macd_signal if macd and macd_signal else None
        support = indicators.get('support')
        resistance = indicators.get('resistance')

        logger.debug(f"[Strategy] Evaluating {trade_type} trade — Entry: {entry_price}, Price: {price}, Confidence: {confidence:.2f}")

        if is_day_trade(position):
            logger.debug("[Strategy] Trade type: Day Trade")

            if is_market_closing_soon(timestamp):
                logger.info("⏰ Market is closing soon — exiting day trade.")
                return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

            # 📉 Smart trailing logic
            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                if price <= price * (1 - 0.10):
                    logger.info("📉 Trailing stop-loss triggered after gain.")
                    return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

            # 🛑 ATR-based stop for day trades
            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER):
                logger.info("🛑 ATR stop-loss hit (day trade).")
                return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

        elif is_swing_trade(position):
            logger.debug("[Strategy] Trade type: Swing Trade")

            # 🛑 ATR-based stop for swing trades
            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER * 1.5):
                logger.info("🛑 ATR stop-loss hit (swing trade).")
                return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

        return action if not RETURN_META_FEEDBACK else (action, meta_feedback)

    except Exception as e:
        logger.error(f"[Strategy Error] {e}")
        logger.debug(traceback.format_exc())
        return "exit" if not RETURN_META_FEEDBACK else ("exit", None)