import csv
import os
from datetime import datetime
import pandas as pd

TRADE_LOG_FILE = "trades.csv"

def log_trade(trade_data: dict):
    file_exists = os.path.isfile(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=trade_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_data)

def log_trade_exit(trade: dict):
    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "trade_id": trade.get("id"),
        "symbol": trade.get("symbol"),
        "type": trade.get("trade_type", "exit"),
        "status": "closed",
        "profit": trade.get("profit", 0.0)
    }

    file_exists = os.path.isfile(TRADE_LOG_FILE)

    with open(TRADE_LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

def get_daily_trade_summary(csv_path=TRADE_LOG_FILE):
    if not os.path.exists(csv_path):
        return "No trade data available today."

    df = pd.read_csv(csv_path)
    if df.empty:
        return "No trades were logged today."

    today = datetime.now().strftime("%Y-%m-%d")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_today = df[df["timestamp"].dt.strftime("%Y-%m-%d") == today]

    if df_today.empty:
        return "No trades executed today."

    num_trades = len(df_today)
    wins = df_today[df_today["profit"] > 0]
    losses = df_today[df_today["profit"] <= 0]
    total_pnl = df_today["profit"].sum()
    win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0

    return (
        f"📅 Date: {today}\n"
        f"🔢 Trades: {num_trades}\n"
        f"✅ Wins: {len(wins)}\n"
        f"❌ Losses: {len(losses)}\n"
        f"📈 Win Rate: {win_rate:.2f}%\n"
        f"💰 Total PnL: ${total_pnl:.2f}"
    )