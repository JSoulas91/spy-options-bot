import requests
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

class TelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text):
        if not self.bot_token or not self.chat_id:
            print("Telegram credentials not set.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(self.api_url, data=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False

    def send_trade_notification(self, trade_result):
        msg = (
            f"📥 *Trade Executed*\n"
            f"🧠 Signal: *{trade_result['type'].upper()}*\n"
            f"🎯 Confidence: *{trade_result['confidence']}*\n"
            f"📈 PnL: *{trade_result['pnl']}%*"
        )
        self.send_message(msg)

    def send_daily_summary(self, trades):
        summary = self.format_trade_summary(trades)
        self.send_message(summary)

    def format_trade_summary(self, trades):
        if not trades:
            return "*📊 Daily Summary*\n\n_No trades executed today._"

        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = len(trades) - wins
        avg_conf = sum(t['confidence'] for t in trades) / len(trades)

        lines = [f"*📊 Daily Summary — {datetime.now().strftime('%Y-%m-%d')}*"]
        lines.append(f"Trades taken: *{len(trades)}*")
        lines.append(f"Wins: *{wins}* | Losses: *{losses}*")
        lines.append(f"Total PnL: *{total_pnl:.2f}%*")
        lines.append(f"Average Confidence: *{avg_conf:.1f}*")
        lines.append("\n*🧾 Trade Details:*")

        for t in trades:
            pnl_color = "🟢" if t['pnl'] > 0 else "🔴"
            lines.append(
                f"{pnl_color} `{t['symbol']} {t['type']}` | Entry: {t['entry']} → Exit: {t['exit']} | "
                f"PnL: `{t['pnl']:.2f}%` | Confidence: `{t['confidence']}`"
            )

        return "\n".join(lines)