import os
import csv
import json
from datetime import datetime

# Paths for data logging
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spy_data.csv")
TRAINING_LOG_PATH = os.path.join(BASE_DIR, "training_log.jsonl")

# Define the feature schema including OHLCV and engineered features
DEFAULT_FEATURES = [
    'open',
    'high',
    'low',
    'volume',
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

# Limit for how many rows to retain
MAX_ROWS = 5000

def log_training_example(timestamp, close, features: dict, label: int = None):
    """
    Appends a single training example to spy_data.csv.
    Ensures consistent header and trims file if it grows too large.
    Also logs JSONL version to training_log.jsonl.
    """
    fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + DEFAULT_FEATURES[4:] + ['label']

    # Build row for CSV
    row = {
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'open': features.get('open', ''),
        'high': features.get('high', ''),
        'low': features.get('low', ''),
        'close': close,
        'volume': features.get('volume', ''),
        'label': label if label is not None else ''
    }

    # Add remaining features
    for key in DEFAULT_FEATURES[4:]:
        row[key] = features.get(key, '')

    file_exists = os.path.exists(DATA_PATH)

    try:
        with open(DATA_PATH, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[logger.py] Failed to write training example to CSV: {e}")
        return

    try:
        _prune_if_necessary(fieldnames)
    except Exception as e:
        print(f"[logger.py] Failed to prune spy_data.csv: {e}")

    # Also log JSONL version
    try:
        log_training_jsonl(timestamp, close, features, label)
    except Exception as e:
        print(f"[logger.py] JSONL logging failed: {e}")


def log_training_jsonl(timestamp, close, features: dict, label: int = None):
    """
    Logs the training data as JSONL to training_log.jsonl for ML tracking/debugging.
    """
    payload = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "close": close,
        "features": features,
        "label": label,
    }
    try:
        with open(TRAINING_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"[logger.py] Failed to write to training_log.jsonl: {e}")


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