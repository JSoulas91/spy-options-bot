# ml/logger.py
import os
import csv
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), "spy_data.csv")

# Define a fixed schema to avoid header mismatches
DEFAULT_FEATURES = [
    'vix', 'rsi', 'macd', 'sma_ratio', 'volume_zscore', 'regime_bull', 'regime_bear'
    # Add all expected feature names used in your system
]

def log_training_example(timestamp, close, features: dict, label: int = None):
    """
    Appends a single training example to spy_data.csv.
    Ensures consistent header to prevent retrain issues.
    """
    fieldnames = ['timestamp', 'close'] + DEFAULT_FEATURES + ['label']
    row = {
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'close': close,
        'label': label if label is not None else ''
    }

    # Fill in features (missing ones default to empty string)
    for key in DEFAULT_FEATURES:
        row[key] = features.get(key, '')

    file_exists = os.path.exists(DATA_PATH)
    with open(DATA_PATH, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)