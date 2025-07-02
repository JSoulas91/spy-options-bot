import os
import csv
import json
from datetime import datetime
import numpy as np

# Paths for data logging
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "spy_data.csv")
TRAINING_LOG_PATH = os.path.join(BASE_DIR, "training_log.jsonl")

# Limit for how many rows to retain
MAX_ROWS = 5000

DEBUG = False  # Set to True to print all logged features


def _safe_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else ''
    return value


def _get_all_feature_keys(features: dict) -> list:
    static = ['vix', 'confidence', 'regime_bull', 'regime_bear']
    dynamic = sorted([k for k in features.keys() if k not in static])
    return static + dynamic


def log_training_example(timestamp, close, features: dict, label: int = None,
                         meta_entry_state: list = None, meta_exit_state: list = None):
    """
    Appends a single training example to spy_data.csv and logs JSONL version.
    """
    if isinstance(timestamp, str):
        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    feature_keys = _get_all_feature_keys(features)
    fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + feature_keys + ['label']

    row = {
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'open': _safe_scalar(features.get('open', '')),
        'high': _safe_scalar(features.get('high', '')),
        'low': _safe_scalar(features.get('low', '')),
        'close': close,
        'volume': _safe_scalar(features.get('volume', '')),
        'label': label if label is not None else ''
    }

    for key in feature_keys:
        row[key] = _safe_scalar(features.get(key, ''))

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

    try:
        log_training_jsonl(timestamp, close, features, label, meta_entry_state, meta_exit_state)
    except Exception as e:
        print(f"[logger.py] JSONL logging failed: {e}")

def log_training_jsonl(timestamp, close, features: dict, label: int = None,
                       meta_entry_state: list = None, meta_exit_state: list = None):
    """
    Logs the training data as JSONL to training_log.jsonl for ML tracking/debugging.
    """
    payload = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp),
        "close": close,
        "features": features,
        "label": label,
        "meta_entry_state": meta_entry_state,
        "meta_exit_state": meta_exit_state,
    }
    try:
        with open(TRAINING_LOG_PATH, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception as e:
        print(f"[logger.py] Failed to write to training_log.jsonl: {e}")


def _prune_if_necessary(fieldnames: list):
    """
    Prunes spy_data.csv to retain only the most recent MAX_ROWS.
    Avoids unbounded file growth.
    """
    if not os.path.exists(DATA_PATH):
        return

    with open(DATA_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) <= MAX_ROWS:
        return

    trimmed = rows[-MAX_ROWS:]
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trimmed)