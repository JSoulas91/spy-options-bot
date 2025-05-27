def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, lower_band
