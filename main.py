import time
import traceback
from scheduler import run_scheduler_loop
from utils.logger import bot_logger
from telegram_bot import send_telegram_message

def main():
    bot_logger.info("🚀 SPY Options Trading Bot is launching...")
    send_telegram_message("🚀 *Bot Online*\nScheduler loop is starting.")

    try:
        run_scheduler_loop()
    except Exception as e:
        bot_logger.critical(f"🔥 Fatal error in scheduler loop: {e}")
        bot_logger.debug(traceback.format_exc())
        send_telegram_message(f"❌ *Fatal Error in main.py*\n\nError: `{str(e)}`")
        raise
    finally:
        bot_logger.info("🛑 Bot has stopped. main.py exiting.")
        send_telegram_message("🛑 *Bot Offline*\nMain loop has exited.")

if __name__ == "__main__":
    main()