import requests
import traceback
from datetime import datetime
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger  # ✅ Logger

class TelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text):
        if not self.bot_token or not self.chat_id:
            bot_logger.error("❌ Telegram credentials not set.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(self.api_url, data=payload, timeout=10)
            if response.status_code == 200:
                bot_logger.info("📨 Telegram message sent successfully.")
                return True
            else:
                bot_logger.warning(f"⚠️ Telegram API error: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.Timeout:
            bot_logger.error("⏱️ Telegram message timeout.")
        except requests.exceptions.RequestException as e:
            bot_logger.error(f"[Telegram Send Error] {str(e)}")
        except Exception as e:
            bot_logger.error(f"[Unexpected Telegram Error] {str(e)}")
        finally:
            bot_logger.debug(traceback.format_exc())

        return False

    def send_trade_notification(self, trade_result):
        try:
            msg = (
                f"📥 *Trade Executed*\n"
                f"🧠 Signal: *{trade_result.get('type', 'N/A').upper()}*\n"
                f"🎯 Confidence: *{trade_result.get('confidence', 'N/A')}*\n"
                f"📈 PnL: *{trade_result.get('pnl', 'N/A')}%*"
            )
            self.send_message(msg)
        except Exception as e:
            bot_logger.error(f"[Trade Notification Error] {str(e)}")
            bot_logger.debug(traceback.format_exc())

    def send_daily_summary(self, trades):
        try:
            summary = self.format_trade_summary(trades)
            self.send_message(summary)
        except Exception as e:
            bot_logger.error(f"[Daily Summary Error] {str(e)}")
            bot_logger.debug(traceback.format_exc())

    def format_trade_summary(self, trades):
        try:
            if not trades:
                return "*📊 Daily Summary*\n\n_No trades executed today._"

            total_pnl = sum(t.get('pnl', 0) for t in trades)
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losses = len(trades) - wins
            avg_conf = sum(t.get('confidence', 0) for t in trades) / len(trades)

            lines = [f"*📊 Daily Summary — {datetime.now().strftime('%Y-%m-%d')}*"]
            lines.append(f"Trades taken: *{len(trades)}*")
            lines.append(f"Wins: *{wins}* | Losses: *{losses}*")
            lines.append(f"Total PnL: *{total_pnl:.2f}%*")
            lines.append(f"Average Confidence: *{avg_conf:.1f}*")
            lines.append("\n*🧾 Trade Details:*")

            for t in trades:
                pnl = t.get('pnl', 0)
                pnl_color = "🟢" if pnl > 0 else "🔴"
                lines.append(
                    f"{pnl_color} `{t.get('symbol', 'N/A')} {t.get('type', 'N/A')}` | "
                    f"Entry: {t.get('entry', '?')} → Exit: {t.get('exit', '?')} | "
                    f"PnL: `{pnl:.2f}%` | Confidence: `{t.get('confidence', '?')}`"
                )

            return "\n".join(lines)

        except Exception as e:
            bot_logger.error(f"[Summary Formatting Error] {str(e)}")
            bot_logger.debug(traceback.format_exc())
            return "*📊 Daily Summary Error*\n\n_An error occurred while formatting the report._"