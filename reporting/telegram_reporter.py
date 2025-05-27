import requests
import json

class TelegramReporter:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send_message(self, text):
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(self.api_url, data=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False

    def send_trade_summary(self, trades):
        if not trades:
            return self.send_message("No trades executed today.")
        msg = "<b>Trade Summary</b>\n"
        for trade in trades:
            msg += f"\n🔹 <b>{trade['symbol']}</b>\n"
            msg += f"Type: {trade['type']}\n"
            msg += f"Entry: {trade['entry_price']}, Exit: {trade['exit_price']}\n"
            msg += f"PnL: {trade['pnl']:.2f}%\n"
        return self.send_message(msg)
