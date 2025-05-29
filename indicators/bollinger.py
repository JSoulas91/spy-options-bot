import pandas as pd
import traceback
from utils.logger import bot_logger

def calculate_bollinger_bands(data, period=20, std_dev=2):
    try:
        if 'close' not in data.columns:
            bot_logger.error("❌ [Bollinger] 'close' column missing in data.")
            return None, None

        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + std_dev * std
        lower_band = sma - std_dev * std
        return upper_band, lower_band

    except Exception as e:
        bot_logger.error(f"[Bollinger Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None, None