import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_macd(data, fast=12, slow=26, signal=9):
    try:
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    except Exception as e:
        bot_logger.error(f"[MACD Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None, None
