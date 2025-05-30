import traceback
from datetime import datetime, time
from helpers import is_day_trade, is_swing_trade
from config import (
    CONFIDENCE_THRESHOLD,
    STOP_LOSS_ATR_MULTIPLIER,
    TRAILING_STOP_PERCENT,
)
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message
from event_filter import is_high_risk_event_active  # ✅ NEW

def is_market_closing_soon(timestamp_str):
    try:
        current_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").time()
        return current_time >= time(15, 55)  # 3:55 PM ET
    except Exception:
        return False

def evaluate_trade(position, market_data):
    """
    Evaluate trade action for current open position.
    :param position: dict of current open position info
    :param market_data: dict of latest price, indicators, confidence, timestamp, etc.
    :return: 'hold', 'exit', or 'scale'
    """
    try:
        action = "hold"
        entry_price = position.get('entry_price')
        price = market_data.get('price')
        indicators = market_data.get("indicators", {})
        confidence = market_data.get("confidence_score", 0)
        timestamp = market_data.get("timestamp", "")

        if entry_price is None or price is None:
            raise ValueError("Missing 'entry_price' or 'price'.")

        # Event-based risk filtering
        if is_high_risk_event_active():
            logger.warning("🚨 Live economic event detected — exiting to reduce risk.")
            send_telegram_message("🚨 *Live Economic Event Detected*\nAuto-exiting position to reduce risk exposure.")
            return "exit"

        # Indicators
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

        # === DAY TRADE LOGIC ===
        if is_day_trade(position):
            logger.debug("[Strategy] Trade type: Day Trade")

            if is_market_closing_soon(timestamp):
                logger.info("⏰ Market closing soon — exiting day trade.")
                return "exit"

            if price >= entry_price * (1 + TRAILING_STOP_PERCENT):
                if price <= price * (1 - 0.10):
                    logger.info("📉 Trailing stop-loss hit after 10% gain.")
                    return "exit"

            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER):
                logger.info("🛑 Day trade ATR stop-loss hit.")
                return "exit"

            if rsi and rsi > 70 and vwap and price < vwap:
                logger.info("⚠️ RSI overbought + VWAP rejection — exit.")
                return "exit"
            elif rsi and rsi < 30 and vwap and price > vwap:
                logger.info("⚠️ RSI oversold + VWAP reclaim — exit.")
                return "exit"

            if price > upper_band or price < lower_band:
                logger.info("⚠️ Price outside Bollinger Bands — exit.")
                return "exit"

            if ema_50 and ema_200 and ema_50 < ema_200:
                logger.info("⚠️ EMA50 below EMA200 — exit.")
                return "exit"

            if resistance and price > resistance * 0.995:
                logger.info("⚠️ Near resistance — consider profit.")
                return "exit"
            if support and price < support * 1.005:
                logger.info("⚠️ Near support — protect capital.")
                return "exit"

            if confidence < CONFIDENCE_THRESHOLD * 0.8:
                logger.info("⚠️ Confidence score dropped significantly — exit.")
                return "exit"

        # === SWING TRADE LOGIC ===
        elif is_swing_trade(position):
            logger.debug("[Strategy] Trade type: Swing Trade")

            if price >= entry_price * (1 + TRAILING_STOP_PERCENT * 1.5):
                if price <= price * (1 - 0.10):
                    logger.info("📉 Swing trailing stop-loss hit after strong gains.")
                    return "exit"

            if atr and price <= entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER * 1.2):
                logger.info("🛑 Swing trade ATR stop-loss hit.")
                return "exit"

            if rsi and macd_hist:
                if rsi > 70 and macd_hist < 0 and price < resistance:
                    logger.info("⚠️ RSI overbought + MACD reversal + resistance — exit.")
                    return "exit"
                elif rsi < 30 and macd_hist > 0 and price > support:
                    logger.info("⚠️ RSI oversold + MACD recovery + support — exit.")
                    return "exit"

            if vwap and price < vwap and rsi and rsi > 65:
                logger.info("⚠️ VWAP fade + elevated RSI — swing exit.")
                return "exit"

            if ema_50 and ema_200 and ema_50 < ema_200:
                logger.info("⚠️ EMA50 crossed below EMA200 — exit.")
                return "exit"

            if price > upper_band or price < lower_band:
                logger.info("⚠️ Bollinger Band extremes — swing exit.")
                return "exit"

            if resistance and price > resistance * 0.995:
                logger.info("⚠️ Approaching resistance — take profit.")
                return "exit"
            if support and price < support * 1.005:
                logger.info("⚠️ Approaching support — risk manage.")
                return "exit"

            if confidence < CONFIDENCE_THRESHOLD * 0.75:
                logger.info("⚠️ Confidence dropped — swing exit filter triggered.")
                return "exit"

        logger.debug(f"[Strategy] Final action: {action}")
        return action

    except Exception as e:
        logger.error(f"[Strategy Error] {str(e)}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"⚠️ *Strategy Module Error*\n"
            f"Could not evaluate trade.\n"
            f"Reason: `{str(e)}`"
        )
        return "hold"