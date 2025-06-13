import os
import csv
from datetime import datetime
from collections import deque

DATA_PATH = os.path.join(os.path.dirname(__file__), "spy_data.csv")

# Define the feature schema
DEFAULT_FEATURES = [
    'vix',
    'rsi',
    'macd',
    'sma_ratio',
    'volume_zscore',
    'regime_bull',
    'regime_bear',
    'confidence',
    'hour',
    'atr',
    'pnl'
]

# Limit for how many rows to retain (oldest removed when exceeded)
MAX_ROWS = 5000

def log_training_example(timestamp, close, features: dict, label: int = None):
    """
    Appends a single training example to spy_data.csv.
    Ensures consistent header and trims file if it grows too large.
    """
    fieldnames = ['timestamp', 'close'] + DEFAULT_FEATURES + ['label']
    row = {
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'close': close,
        'label': label if label is not None else ''
    }

    for key in DEFAULT_FEATURES:
        row[key] = features.get(key, '')

    file_exists = os.path.exists(DATA_PATH)

    try:
        with open(DATA_PATH, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[logger.py] Failed to write training example: {e}")
        return

    try:
        _prune_if_necessary(fieldnames)
    except Exception as e:
        print(f"[logger.py] Failed to prune spy_data.csv: {e}")


def _prune_if_necessary(fieldnames: list):
    """
    Prunes spy_data.csv to retain only the most recent MAX_ROWS.
    Avoids unbounded file growth.
    """
    with open(DATA_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) <= MAX_ROWS:
        return

    trimmed = rows[-MAX_ROWS:]
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trimmed)