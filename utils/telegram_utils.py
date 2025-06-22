import os
import io
import re
import requests
import matplotlib.pyplot as plt

def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters.
    """
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

def send_telegram_message(text: str, image_bytes: bytes = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return

    # Escape for MarkdownV2
    safe_text = escape_markdown_v2(text)

    if image_bytes:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {'photo': ('reward_chart.png', image_bytes)}
        data = {
            'chat_id': chat_id,
            'caption': safe_text,
            'parse_mode': 'MarkdownV2'
        }
        try:
            response = requests.post(url, files=files, data=data)
            if response.status_code != 200:
                print("❌ Failed to send Telegram image message:", response.text)
        except Exception as e:
            print("❌ Telegram image send error:", str(e))
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': safe_text,
            'parse_mode': 'MarkdownV2'
        }
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                print("❌ Failed to send Telegram message:", response.text)
        except Exception as e:
            print("❌ Telegram send error:", str(e))

def send_plot(plt_obj, caption="Meta‑Agent Reward Plot"):
    """
    Send a matplotlib plot to Telegram as an image.
    """
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    send_telegram_message(caption, image_bytes=buf.read())