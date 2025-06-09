from dotenv import load_dotenv
import os
load_dotenv()

from utils.telegram_utils import send_telegram_message

send_telegram_message("✅ Telegram is working!")
