# utils/trade_history.py

import os
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import bot_logger
from config import META_STATE_LOOKBACK_MINUTES

TRADE_HISTORY_CSV = os.getenv("TRADE_HISTORY_CSV", "data/trade_history.csv")

def get_recent_trade_results(lookback_minutes=META_STATE_LOOKBACK_MINUTES):
    """
    Load recent trades and return a list of outcomes ['win', 'loss', 'neutral']
    within the specified lookback period.
    """
    try:
        if not os.path.exists(TRADE_HISTORY_CSV):
            bot_logger.warning(f"No trade history file found at {TRADE_HISTORY_CSV}")
            return []

        df = pd.read_csv(TRADE_HISTORY_CSV)

        if 'timestamp' not in df.columns or 'outcome' not in df.columns:
            bot_logger.error("Trade history CSV must contain 'timestamp' and 'outcome' columns.")
            return []

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_df = df[df['timestamp'] >= cutoff]

        if recent_df.empty:
            bot_logger.info("No recent trades found in the lookback window.")
            return []

        outcomes = recent_df['outcome'].str.lower().tolist()
        bot_logger.info(f"📊 Loaded {len(outcomes)} recent trade outcomes: {outcomes}")
        return outcomes

    except Exception as e:
        bot_logger.exception("Failed to load recent trade results.")
        return []