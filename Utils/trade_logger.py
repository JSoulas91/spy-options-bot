import csv
import os
from datetime import datetime

def log_trade(trade_data, filename="trades.csv"):
    fieldnames = ["timestamp", "trade_type", "option_symbol", "entry_price", "exit_price", "profit_loss", "confidence", "notes"]

    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_type": trade_data.get("type", ""),
            "option_symbol": trade_data.get("symbol", ""),
            "entry_price": trade_data.get("entry", ""),
            "exit_price": trade_data.get("exit", ""),
            "profit_loss": trade_data.get("pnl", ""),
            "confidence": trade_data.get("confidence", ""),
            "notes": trade_data.get("notes", "")
        })
