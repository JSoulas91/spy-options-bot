# strategy.py

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
from event_filter import is_high_risk_event_active, has_monday_event
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

def get_adaptive_confidence_threshold(meta_state=None):
    base_threshold = CONFIDENCE_THRESHOLD

    if ENABLE_VIX_THROTTLING or ENABLE_ADAPTIVE_CONFIDENCE:
        try:
            vix_value = get_current_vix()
            logger.info(f"📈 VIX value: {vix_value}")
            if vix_value is None:
                return base_threshold
            if vix_value >= VIX_MAX_THRESHOLD:
                logger.warning("❌ VIX above max threshold — skipping trade.")
                return float("inf")
            elif vix_value >= VIX_MODERATE_THRESHOLD:
                adjusted_threshold = base_threshold + CONFIDENCE_STEP_UP
                logger.info(f"⚠️ Elevated VIX — adjusting confidence threshold to {adjusted_threshold}")
                return adjusted_threshold
        except Exception as e:
            logger.error(f"[VIX Error] Failed to retrieve VIX: {str(e)}")
    return base_threshold

def evaluate_trade(position, market_data):
    try:
        action = "hold"
        entry_price = position.get('entry_price')
        price = market_data.get('price')
        indicators = market_data.get("indicators", {})
        confidence = market_data.get("confidence_score", 0)
        timestamp = market_data.get("timestamp", "")

        if entry_price is None or price is None:
            raise ValueError("Missing 'entry_price' or 'price'.")

        if is_high_risk_event_active():
            logger.warning("🚨 Live economic event detected — exiting to reduce risk.")
            send_telegram_message("🚨 *Live Economic Event Detected*\nAuto-exiting position to reduce risk exposure.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", None)

        vix = get_current_vix()
        trade_type = "day" if is_day_trade(position) else "swing"
        hour = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").hour if timestamp else 12
        volatility = indicators.get("atr", 0)

        meta_state = [confidence, vix or 0, hour, 0 if trade_type == "day" else 1, volatility]
        meta_action = meta_agent.select_action(meta_state)

        logger.info(f"🧠 Meta-State: {meta_state} | Meta-Action: {meta_action}")

        meta_feedback = {"meta_state": meta_state, "meta_action": meta_action}

        if meta_action == 0:
            logger.info("🧠 Meta-Agent action: skip — confidence not met.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)
        elif meta_action == 1:
            logger.info("🧠 Meta-Agent action: hold — continue evaluation.")
        elif meta_action == 2:
            logger.info("🧠 Meta-Agent action: force exit — exiting position.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

        adaptive_threshold = get_adaptive_confidence_threshold(meta_state)
        if confidence < adaptive_threshold:
            logger.info(f"⚠️ Confidence {confidence:.2f} below adaptive threshold {adaptive_threshold:.2f} — exiting.")
            return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

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

        logger.debug(f"[Strategy] Evaluating trade — Entry: {entry_price}, Current: {price}, Confidence: {confidence}")

        if is_day_trade(position):
            logger.debug("[Strategy] Trade type: Day Trade")

            if is_market_closing_soon(timestamp):
                logger.info("⏰ Market closing soon — exiting day trade.")
                return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                if price <= price * (1 - 0.10):
                    logger.info("📉 Trailing stop-loss hit after 10% gain.")
                    return "exit" if not RETURN_META_FEEDBACK else ("exit", meta_feedback)

            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER):
                logger.info("🛑 Day trade ATR stop 