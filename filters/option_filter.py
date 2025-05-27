def filter_options_by_greeks(options, min_delta=0.2, max_delta=0.8, max_theta=-0.01, max_vega=0.3, max_gamma=0.2):
    filtered = []
    for option in options:
        delta = option.get('delta', 0)
        theta = option.get('theta', 0)
        vega = option.get('vega', 0)
        gamma = option.get('gamma', 0)
        if min_delta <= abs(delta) <= max_delta and theta >= max_theta and vega <= max_vega and gamma <= max_gamma:
            filtered.append(option)
    return filtered

def filter_by_liquidity(options, min_volume=100, min_open_interest=100):
    return [opt for opt in options if opt.get('volume', 0) >= min_volume and opt.get('open_interest', 0) >= min_open_interest]
