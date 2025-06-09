# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

print("BOT TOKEN:", os.getenv("TELEGRAM_BOT_TOKEN"))
print("CHAT ID:", os.getenv("TELEGRAM_CHAT_ID"))
