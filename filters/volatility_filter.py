# volatility_filter.py

from config import VIX_CAUTION_THRESHOLD, VIX_EXTREME_THRESHOLD
from utils.logger import bot_logger as logger

def check_volatility(vix_value):
    """
    Analyze current VIX value and return volatility status.
    :param vix_value: float — current value of the VIX (CBOE Volatility Index)
    :return: 'normal', 'high', or 'extreme'
    """
    if vix_value is None:
        logger.warning("⚠️ VIX value missing — cannot evaluate volatility.")
        return 'unknown'

    try:
        vix = float(vix_value)

        if vix >= VIX_EXTREME_THRESHOLD:
            logger.warning(f"🚨 Extreme volatility detected (VIX={vix}) — throttle or halt trades.")
            return 'extreme'
        elif vix >= VIX_CAUTION_THRESHOLD:
            logger.info(f"⚠️ Elevated volatility detected (VIX={vix}) — reduce trade size or risk.")
            return 'high'
        else:
            logger.debug(f"✅ Normal volatility (VIX={vix}) — trading conditions stable.")
            return 'normal'

    except Exception as e:
        logger.error(f"[Volatility Filter Error] {str(e)}")
        return 'unknown'

def is_high_volatility(vix_value):
    """
    Returns True if VIX is above the caution threshold.
    """
    status = check_volatility(vix_value)
    return status in ['high', 'extreme']

def is_extreme_volatility(vix_value):
    """
    Returns True only if VIX is above the extreme threshold.
    """
    status = check_volatility(vix_value)
    return status == 'extreme'