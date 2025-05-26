import requests
import os
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def format_trade_summary(trades):
    """
    Accepts a list of trade dictionaries:
    [
        {
            'symbol': 'SPY',
            'type': 'CALL',
            'entry': 1.45,
            'exit': 1.70,
            'confidence': 84,
            'pnl': 17.2
        },
        ...
    ]
    Returns a formatted message string.
    """
    if not trades:
        return "*📊 Daily Summary*\n\n_No trades executed today._"

    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = len(trades) - wins
    avg_conf = sum(t['confidence'] for t in trades) / len(trades)

    lines = [f"*📊 Daily Summary — {datetime.now().strftime('%Y-%m-%d')}*"]
    lines.append(f"Trades taken: {len(trades)}")
    lines.append(f"Wins: {wins} | Losses: {losses}")
    lines.append(f"Total PnL: `{total_pnl:.2f}%`")
    lines.append(f"Average Confidence: `{avg_conf:.1f}`")
    lines.append("\n*🧾 Trade Details:*")

    for t in trades:
        pnl_color = "🟢" if t['pnl'] > 0 else "🔴"
        lines.append(
            f"{pnl_color} `{t['symbol']} {t['type']}` | Entry: {t['entry']} → Exit: {t['exit']} | "
            f"PnL: `{t['pnl']:.2f}%` | Confidence: `{t['confidence']}`"
        )

    return "\n".join(lines)

def send_daily_summary(trades):
    summary = format_trade_summary(trades)
    send_telegram_message(summary)
