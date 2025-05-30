from datetime import datetime
import config
from utils.logger import bot_logger

def score_option(opt, underlying_price=None):
    """Score options based on greeks, liquidity, spread, and proximity to ATM."""
    score = 0

    delta = abs(opt.get('delta', 0))
    theta = opt.get('theta', 0)
    vega = opt.get('vega', 0)
    gamma = opt.get('gamma', 0)
    volume = opt.get('volume', 0)
    open_interest = opt.get('open_interest', 0)
    bid = opt.get('bid', 0)
    ask = opt.get('ask', 0)
    strike = opt.get('strike', 0)

    if bid <= 0 or ask <= 0:
        return -1  # invalid pricing

    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid if mid else 1

    # Moneyness proximity scoring
    if underlying_price and strike:
        moneyness = abs(strike - underlying_price)
        score -= moneyness * 2  # penalize OTM/ITM

    score += delta * 5
    score += -vega * 2
    score += -gamma * 2
    score += theta * 3
    score += volume * 0.01
    score += open_interest * 0.01
    score += -spread_pct * 10

    return round(score, 2)


def filter_options(options, underlying_price=None, vix=None, top_n=10, directional_bias=None):
    """
    Filters and scores options based on greeks, liquidity, expiry, moneyness, spread, VIX, and directional skew.
    """

    filtered = []
    today = datetime.utcnow().date()
    reasons_logged = 0

    for opt in options:
        delta = abs(opt.get('delta', 0))
        theta = opt.get('theta', 0)
        vega = opt.get('vega', 0)
        gamma = opt.get('gamma', 0)
        volume = opt.get('volume', 0)
        open_interest = opt.get('open_interest', 0)
        bid = opt.get('bid', 0)
        ask = opt.get('ask', 0)
        strike = opt.get('strike', 0)
        expiry = opt.get('expiry_date')
        otype = opt.get('type', '').lower()

        # --- Expiry Filter ---
        if expiry is None:
            continue
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
        days_to_expiry = (expiry - today).days
        if days_to_expiry < config.MIN_EXPIRY_DAYS:
            if config.DEBUG_OPTION_FILTER:
                bot_logger.debug(f"❌ Rejected: {otype} {strike} - Expiry too soon: {days_to_expiry}d")
            continue

        # --- Option Type Filter ---
        if config.OPTION_TYPE_FILTER != "both" and otype != config.OPTION_TYPE_FILTER:
            continue

        # --- Greeks Filter ---
        if not (config.MIN_DELTA <= delta <= config.MAX_DELTA):
            if config.DEBUG_OPTION_FILTER:
                bot_logger.debug(f"❌ Rejected: {otype} {strike} - Delta {delta}")
            continue
        if theta < config.MAX_THETA or vega > config.MAX_VEGA or gamma > config.MAX_GAMMA:
            if config.DEBUG_OPTION_FILTER:
                bot_logger.debug(f"❌ Rejected: {otype} {strike} - Theta/Vega/Gamma")
            continue

        # --- Liquidity Filter ---
        if volume < config.MIN_VOLUME or open_interest < config.MIN_OPEN_INTEREST:
            if config.DEBUG_OPTION_FILTER:
                bot_logger.debug(f"❌ Rejected: {otype} {strike} - Liquidity: V={volume}, OI={open_interest}")
            continue

        # --- Bid-Ask Spread Filter ---
        if bid <= 0 or ask <= 0:
            continue
        spread_pct = (ask - bid) / ((ask + bid) / 2)
        vix_threshold = config.MAX_BID_ASK_SPREAD_PCT
        if config.ENABLE_VIX_THROTTLING and vix:
            if vix > config.VIX_MODERATE_THRESHOLD:
                vix_threshold += 0.02
            if vix > config.VIX_MAX_THRESHOLD:
                vix_threshold += 0.04
        if spread_pct > vix_threshold:
            if config.DEBUG_OPTION_FILTER:
                bot_logger.debug(f"❌ Rejected: {otype} {strike} - Spread too wide: {spread_pct:.2%}")
            continue

        # --- Moneyness Filter with Skew ---
        if underlying_price and strike > 0:
            moneyness = strike / underlying_price
            min_m, max_m = config.MIN_MONEYNESS, config.MAX_MONEYNESS

            if config.ENABLE_DIRECTIONAL_SKEW and directional_bias:
                if directional_bias == 'bullish':
                    max_m += 0.05  # allow further OTM calls
                elif directional_bias == 'bearish':
                    min_m -= 0.05  # allow further OTM puts

            if not (min_m <= moneyness <= max_m):
                if config.DEBUG_OPTION_FILTER:
                    bot_logger.debug(f"❌ Rejected: {otype} {strike} - Moneyness: {moneyness:.2f}")
                continue

        opt['score'] = score_option(opt, underlying_price)
        filtered.append(opt)

    # Rank & return top N contracts
    filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
    return filtered[:top_n]