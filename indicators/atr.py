import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_atr(data, period=14):
    try:
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    except Exception as e:
        bot_logger.error(f"[ATR Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None
