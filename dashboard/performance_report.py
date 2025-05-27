import json
import datetime
import requests

def format_performance_summary(trades):
    """
    trades = list of dictionaries like:
    [
        {'type': 'call', 'entry_price': 1.2, 'exit_price': 1.6, 'pnl': 0.4, 'timestamp': '...'},
        ...
    ]
    """
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    net_pnl = sum(t['pnl'] for t in trades)

    return (
        f"📈 *SPY Options Bot Summary* 📊\n"
        f"• Trades: {total_trades}\n"
        f"• Wins: {wins}\n"
        f"• Losses: {losses}\n"
        f"• Win Rate: {win_rate:.2f}%\n"
        f"• Net PnL: {net_pnl:.2f}\n"
        f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

def send_to_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=data)
    return response.status_code == 200
