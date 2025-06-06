# utils/telegram_notifier.py

import requests
import time
import subprocess
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger

class TelegramNotifier:
    def __init__(self, bot_token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.last_update_id = None  # Tracks last processed message

    def send_message(self, message: str):
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            bot_logger.info("📨 Telegram alert sent.")
        except requests.exceptions.RequestException as e:
            bot_logger.warning(f"[Telegram Error] {e}")

    def send_swing_hold_alert(self, contract: dict, reason: str):
        try:
            symbol = contract.get("symbol", "UNKNOWN")
            expiry = contract.get("expiry", "N/A")
            strike = contract.get("strike", "N/A")
            ctype = contract.get("type", "call").upper()
            entry_price = contract.get("entry_price", "N/A")
            confidence = round(contract.get("confidence", 0.0), 2)

            message = (
                f"📈 <b>Swing Trade Held Over Weekend</b>\n\n"
                f"<b>Contract:</b> {symbol} {expiry} {strike} {ctype}\n"
                f"<b>Entry Price:</b> ${entry_price}\n"
                f"<b>Confidence:</b> {confidence}\n"
                f"<b>Reason:</b> {reason}\n\n"
                f"✅ Meets criteria:\n• VIX OK\n• High Confidence\n• No Monday Events"
            )
            self.send_message(message)
        except Exception as e:
            bot_logger.warning(f"[Swing Alert Error] {e}")

    def listen_for_commands(self):
        logger = bot_logger
        logger.info("📡 Listening for Telegram commands...")

        while True:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {"timeout": 10}
                if self.last_update_id:
                    params["offset"] = self.last_update_id + 1

                response = requests.get(url, params=params)
                response.raise_for_status()
                updates = response.json().get("result", [])

                for update in updates:
                    self.last_update_id = update["update_id"]
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = str(message.get("chat", {}).get("id"))

                    if chat_id != str(self.chat_id):
                        logger.info("⚠️ Ignored message from unauthorized user.")
                        continue

                    if text == "/shutdown_bot":
                        logger.info("🛑 Shutdown command received.")
                        self.send_message("🛑 Shutting down bot...")
                        subprocess.run(["pkill", "-f", "live_runner.py"])
                    elif text == "/restart_bot":
                        logger.info("🔁 Restart command received.")
                        self.send_message("🔁 Restarting bot...")
                        subprocess.Popen(["python3", "live_runner.py"])

            except Exception as e:
                logger.warning(f"[Command Listener Error] {e}")

            time.sleep(3)