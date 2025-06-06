# utils/regime_classifier.py

import pandas as pd
import numpy as np

def classify_regime(spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
    """
    Classify the current market regime based on SPY and VIX.

    Returns:
        "bull", "bear", or "volatility_cluster"
    """
    # Ensure both datasets are aligned by date
    df = spy_data[['close']].rename(columns={'close': 'spy_close'}).copy()
    df['vix_close'] = vix_data['close']
    df.dropna(inplace=True)

    # Calculate rolling returns for SPY (10-day)
    df['spy_return'] = df['spy_close'].pct_change(10)

    # Calculate 5-day and 10-day VIX averages
    df['vix_mean_5'] = df['vix_close'].rolling(5).mean()
    df['vix_mean_10'] = df['vix_close'].rolling(10).mean()

    latest = df.iloc[-1]
    spy_ret = latest['spy_return']
    vix_lvl = latest['vix_close']
    vix_trend = latest['vix_mean_5'] - latest['vix_mean_10']

    # Classification rules
    if spy_ret > 0.02 and vix_lvl < 18 and vix_trend <= 0:
        return "bull"
    elif spy_ret < -0.02 and vix_lvl > 22 and vix_trend >= 0:
        return "bear"
    else:
        return "volatility_cluster"