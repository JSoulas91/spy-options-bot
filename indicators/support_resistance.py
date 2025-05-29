import pandas as pd
import numpy as np
import traceback
from utils.logger import bot_logger

def detect_support_resistance(data, window=5):
    """
    Identify support and resistance levels based on local pivot highs/lows.
    Returns two lists: support_levels, resistance_levels
    """
    support = []
    resistance = []

    try:
        for i in range(window, len(data) - window):
            low_range = data['low'][i - window:i + window + 1]
            high_range = data['high'][i - window:i + window + 1]

            if data['low'][i] == low_range.min():
                support.append((i, data['low'][i]))
            if data['high'][i] == high_range.max():
                resistance.append((i, data['high'][i]))

        return support, resistance

    except Exception as e:
        bot_logger.error(f"[S/R Detection Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return [], []
