# utils/telegram_utils.py
from utils.telegram_notifier import TelegramNotifier

_notifier = TelegramNotifier()

def send_telegram_message(msg: str):
    """Single‑line helper used across the code‑base."""
    _notifier.send_message(msg)