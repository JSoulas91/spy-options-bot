import os
import requests

def send_telegram_message(text: str, image_bytes: bytes = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return

    if image_bytes:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {'photo': ('reward_chart.png', image_bytes)}
        data = {'chat_id': chat_id, 'caption': text, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(url, files=files, data=data)
            if response.status_code != 200:
                print("❌ Failed to send Telegram image message:", response.text)
        except Exception as e:
            print("❌ Telegram image send error:", str(e))
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print("❌ Failed to send Telegram message:", response.text)
        except Exception as e:
            print("❌ Telegram send error:", str(e))