import schedule
import time
import traceback
import pytz
from datetime import datetime
from retrain import retrain_model
from trade_manager import run_trading_bot
from telegram_bot import send_telegram_message
from performance_summary import send_daily_summary
from cleanup import cleanup_logs_and_backups
from utils.logger import bot_logger as logger

# Set Eastern Timezone
eastern = pytz.timezone('US/Eastern')

def log_and_notify(task_name, func):
    start_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
    logger.info(f"[Scheduler] Starting task: {task_name} at {start_time}")
    send_telegram_message(f"⏰ *Scheduled Task Started*\n🛠️ Task: {task_name}\n🕒 Time: {start_time}")
    
    try:
        func()
        logger.info(f"[Scheduler] Task complete: {task_name}")
        send_telegram_message(f"✅ *Task Complete*\n🛠️ {task_name} finished successfully.")
    except Exception as e:
        error_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
        error_msg = (
            f"🚨 *Scheduled Task Error*\n"
            f"🧩 Task: {task_name}\n"
            f"❌ Reason: `{str(e)}`\n"
            f"🕒 Time: {error_time}"
        )
        logger.critical(f"[Scheduler] Task failed: {task_name} — Reason: {e}")
        logger.debug(traceback.format_exc())
        send_telegram_message(error_msg)

# Daily Schedule (Eastern Time)
schedule.every().day.at("09:30").do(lambda: log_and_notify("Trading Bot", run_trading_bot))
schedule.every().day.at("16:45").do(lambda: log_and_notify("Performance Summary", send_daily_summary))
schedule.every().day.at("16:50").do(lambda: log_and_notify("Model Retraining", retrain_model))
schedule.every().day.at("17:00").do(lambda: log_and_notify("Log & Backup Cleanup", cleanup_logs_and_backups))

def run_scheduler():
    logger.info("📅 Scheduler started. Waiting for scheduled tasks...")
    send_telegram_message("✅ *Scheduler Online*\n📆 All tasks initialized and standing by.")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run_scheduler()