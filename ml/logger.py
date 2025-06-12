# ml/logger.py
import os
import csv
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), "spy_data.csv")

def log_training_example(timestamp, close, features: dict, label: int = None):
    """
    Appends a single training example to spy_data.csv.
    Creates file and header if not present.
    """
    fieldnames = ['timestamp', 'close'] + list(features.keys()) + (['label'] if label is not None else [])
    row = {
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'close': close,
        **features
    }
    if label is not None:
        row['label'] = label

    file_exists = os.path.exists(DATA_PATH)
    with open(DATA_PATH, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)