def calculate_vwap(data):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap
