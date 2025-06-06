# monitor/telegram_listener.py

from utils.telegram_notifier import TelegramNotifier

if __name__ == "__main__":
    notifier = TelegramNotifier()
    notifier.listen_for_commands()