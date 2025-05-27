def is_uptrend(data, short_ema, long_ema):
    return short_ema.iloc[-1] > long_ema.iloc[-1] and data['close'].iloc[-1] > short_ema.iloc[-1]

def is_downtrend(data, short_ema, long_ema):
    return short_ema.iloc[-1] < long_ema.iloc[-1] and data['close'].iloc[-1] < short_ema.iloc[-1]

def detect_support_resistance(data, window=10):
    support = []
    resistance = []
    for i in range(window, len(data) - window):
        low_range = data['low'][i - window:i + window]
        high_range = data['high'][i - window:i + window]
        current_low = data['low'][i]
        current_high = data['high'][i]

        if current_low == min(low_range):
            support.append((i, current_low))
        if current_high == max(high_range):
            resistance.append((i, current_high))

    return support, resistance

def confirm_trend_with_higher_timeframe(short_tf_trend, long_tf_trend):
    return short_tf_trend == long_tf_trend
