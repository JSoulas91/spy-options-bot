import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_vwap(data):
    try:
        for col in ['high', 'low', 'close', 'volume']:
            if col not in data.columns:
                bot_logger.error(f"❌ [VWAP] '{col}' column missing in data.")
                return None

        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap

    except Exception as e:
        bot_logger.error(f"[VWAP Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None