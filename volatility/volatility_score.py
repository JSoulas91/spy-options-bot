def compute_volatility_score(atr, close):
    if atr is None or close is None or close == 0:
        return 0
    return round((atr / close) * 100, 2)
