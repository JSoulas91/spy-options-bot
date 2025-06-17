import numpy as np
import pandas as pd

REQUIRED_FEATURE_COLUMNS = [
    'confidence', 'setup_quality', 'vix', 'realized_vol', 'trade_type', 'total_signals_today',
    'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'vwap', 'atr_14', 'adx_14'
]

def build_features_for_trade(meta_log_entry: dict) -> pd.DataFrame:
    """
    Accepts a dict from meta_log.jsonl (or simulation) and returns a single-row DataFrame
    with all required features for classifier inference.
    Missing features are filled with np.nan.
    """
    row = {}
    for col in REQUIRED_FEATURE_COLUMNS:
        row[col] = meta_log_entry.get(col, np.nan)
    return pd.DataFrame([row])