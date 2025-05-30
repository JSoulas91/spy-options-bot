import schedule
import time
import traceback
import pytz
from datetime import datetime
from strategy.strategy import run_strategy  # ✅ updated path
from retrain import retrain_model
from performance_summary import send_daily_summary
from cleanup import cleanup_logs_and_backups
from telegram_bot import send_telegram_message
from utils.logger import bot_logger as logger

# Timezone
eastern = pytz.timezone('US/Eastern')

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

def retry_task(task_name, func):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[{task_name}] Attempt {attempt} starting...")
            func()
            logger.info(f"[{task_name}] Success on attempt {attempt}")
            return True
        except Exception as e:
            logger.warning(f"[{task_name}] Attempt {attempt} failed: {e}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                f"⚠️ *Retry {attempt}/{MAX_RETRIES} Failed*\n"
                f"🛠️ Task: {task_name}\n"
                f"❌ Reason: `{str(e)}`"
            )
            time.sleep(RETRY_DELAY)
    return False

def log_and_notify(task_name, func):
    start_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
    logger.info(f"📌 [Scheduler] Starting task: {task_name} at {start_time}")
    send_telegram_message(
        f"⏰ *Scheduled Task Started*\n"
        f"🛠️ Task: {task_name}\n"
        f"🕒 Time: {start_time}"
    )

    success = retry_task(task_name, func)

    if success:
        end_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
        logger.info(f"✅ [Scheduler] Task complete: {task_name} at {end_time}")
        send_telegram_message(
            f"✅ *Task Complete*\n"
            f"🛠️ {task_name} finished successfully.\n"
            f"🕒 Time: {end_time}"
        )
    else:
        error_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
        logger.critical(f"❌ [Scheduler] Task FAILED after {MAX_RETRIES} attempts: {task_name}")
        send_telegram_message(
            f"🚨 *Task FAILED*\n"
            f"🧩 Task: {task_name}\n"
            f"❌ All {MAX_RETRIES} retries failed.\n"
            f"🕒 Time: {error_time}"
        )

# ===========================
# ✅ Scheduled Daily Tasks
# ===========================

# Day trades at 9:30 AM ET
schedule.every().day.at("09:30").do(lambda: log_and_notify("Day Trade Strategy", lambda: run_strategy(mode="day")))

# Optional swing trades near close
schedule.every().day.at("15:55").do(lambda: log_and_notify("Swing Trade Strategy", lambda: run_strategy(mode="swing")))

# Daily summary report at 4:45 PM ET
schedule.every().day.at("16:45").do(lambda: log_and_notify("Performance Summary", send_daily_summary))

# Model retraining at 4:50 PM ET
schedule.every().day.at("16:50").do(lambda: log_and_notify("Model Retraining", retrain_model))

# Cleanup logs and backups at 5:00 PM ET
schedule.every().day.at("17:00").do(lambda: log_and_notify("Log & Backup Cleanup", cleanup_logs_and_backups))

# ===========================
# Scheduler Loop
# ===========================

def run_scheduler():
    logger.info("📅 [Scheduler] Started. Waiting for scheduled tasks...")

    startup_time = datetime.now(eastern).strftime("%Y-%m-%d %I:%M %p")
    send_telegram_message(
        f"🔄 *Bot Restart Detected*\n"
        f"🤖 Scheduler loop reloaded.\n"
        f"🕒 Time: {startup_time}"
    )

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.critical(f"🔥 [Scheduler Loop Error] {str(e)}")
            logger.debug(traceback.format_exc())
            send_telegram_message(
                f"🚨 *Fatal Scheduler Loop Error*\n"
                f"Reason: `{str(e)}`"
            )

if __name__ == "__main__":
    run_scheduler()