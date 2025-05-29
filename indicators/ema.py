import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_ema(data, period):
    try:
        if 'close' not in data.columns:
            bot_logger.error("❌ [EMA] 'close' column missing in data.")
            return None

        return data['close'].ewm(span=period, adjust=False).mean()

    except Exception as e:
        bot_logger.error(f"[EMA Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None