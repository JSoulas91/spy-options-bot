import requests
import os

class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message):
        if not self.token or not self.chat_id:
            print("⚠️ Telegram credentials not set in environment variables.")
            return

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"  # Can change to "Markdown" if needed
        }

        try:
            response = requests.post(self.api_url, data=payload)
            if response.status_code != 200:
                print(f"⚠️ Telegram error: {response.text}")
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")