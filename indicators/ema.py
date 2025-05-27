def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()
