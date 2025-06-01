import csv
from datetime import datetime
import os

def log_trade(trade_data):
    """
    trade_data should be a dict with:
    - timestamp
    - action ("buy"/"sell")
    - trade_type ("day" or "swing")
    - symbol
    - profit_loss
    - trade_duration (in minutes)
    - confidence_score
    - indicators: dictionary or summary string
    """
    file_path = 'logs/trades.csv'
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'timestamp',
            'action',
            'trade_type',
            'symbol',
            'profit_loss',
            'trade_duration',
            'confidence_score',
            'indicators'
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow(trade_data)