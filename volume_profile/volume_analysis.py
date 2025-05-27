import numpy as np
import pandas as pd

def detect_volume_spike(data, threshold_multiplier=2):
    avg_volume = data['volume'].rolling(window=20).mean()
    volume_spike = data['volume'] > (avg_volume * threshold_multiplier)
    return volume_spike

def calculate_vwap(data):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    return vwap

def check_price_vs_vwap(data):
    vwap = calculate_vwap(data)
    latest_price = data['close'].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    return latest_price > latest_vwap  # returns True if price is above VWAP
