import requests, traceback, io
from config      import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger

class TelegramBot:
    def __init__(self):
        self.token   = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.base    = f"https://api.telegram.org/bot{self.token}"

    # ── helpers ─────────────────────────────────────────────
    def _post(self, method: str, data=None, files=None):
        url = f"{self.base}/{method}"
        try:
            r = requests.post(url, data=data, files=files, timeout=10)
            if r.status_code == 200:
                bot_logger.debug(f"Telegram {method} ok.")
            else:
                bot_logger.warning(f"Telegram {method} error {r.status_code} {r.text}")
        except Exception as e:
            bot_logger.error(f"Telegram error: {e}")
            bot_logger.debug(traceback.format_exc())

    # ── public API ──────────────────────────────────────────
    def send_message(self, text: str, parse="Markdown"):
        self._post("sendMessage", data={
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse,
        })

    def send_photo(self, image_bytes: bytes, caption: str = ""):
        files = {"photo": ("reward.png", io.BytesIO(image_bytes))}
        data  = {"chat_id": self.chat_id, "caption": caption}
        self._post("sendPhoto", data=data, files=files)

    # existing trade‑notification helpers unchanged …