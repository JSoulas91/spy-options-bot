import time
import traceback
from scheduler import run_scheduler_loop
from utils.logger import bot_logger as logger
from telegram_bot import send_telegram_message

def main():
    logger.info("🚀 SPY Options Trading Bot is launching...")
    send_telegram_message("🚀 *Bot Online*\nMain loop is starting and will self-heal on failure.")

    while True:
        try:
            run_scheduler_loop()  # Infinite loop unless it crashes
        except Exception as e:
            logger.critical(f"🔥 Scheduler loop crashed: {e}")
            logger.debug(traceback.format_exc())

            send_telegram_message(
                f"❌ *Main Loop Crash Detected*\n"
                f"📛 Error: `{str(e)}`\n"
                f"⏳ Retrying in 5 seconds..."
            )
            logger.debug("🔁 Retrying scheduler loop in 5 seconds...")
            time.sleep(5)

        else:
            logger.warning("⚠️ Scheduler loop exited unexpectedly.")
            send_telegram_message("⚠️ *Scheduler exited unexpectedly.* Retrying in 5 seconds...")
            logger.debug("🔁 Retrying scheduler loop due to unexpected exit...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"💀 Bot main entry failed: {e}")
        logger.debug(traceback.format_exc())
        send_telegram_message(f"💥 *Fatal Error in main.py*\n\n{str(e)}")