import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_rsi(data, period=14):
    try:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        bot_logger.error(f"[RSI Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None
