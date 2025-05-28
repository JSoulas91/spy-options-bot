import schedule
import time
from datetime import datetime
from config import (
    TRADE_EXECUTION_TIME,
    SUMMARY_REPORT_TIME,
    RETRAINING_TIME,
    TELEGRAM_ALERTS_ENABLED
)

from trade_manager import run_daily_trades
from telegram_bot import send_daily_summary
from retrain import run_retraining_pipeline
from utils import cleanup_old_logs, backup_logs


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def job_run_trades():
    log("Running daily trade execution.")
    run_daily_trades()


def job_send_summary():
    if TELEGRAM_ALERTS_ENABLED:
        log("Sending daily Telegram performance summary.")
        send_daily_summary()
    else:
        log("Telegram alerts disabled in config.")


def job_retrain_and_cleanup():
    log("Running model retraining and cleanup pipeline.")
    run_retraining_pipeline()
    cleanup_old_logs()
    backup_logs()


def schedule_all_tasks():
    schedule.every().day.at(TRADE_EXECUTION_TIME).do(job_run_trades)
    schedule.every().day.at(SUMMARY_REPORT_TIME).do(job_send_summary)
    schedule.every().day.at(RETRAINING_TIME).do(job_retrain_and_cleanup)


def run_scheduler_loop():
    log("Scheduler started. Waiting for jobs to run...")
    schedule_all_tasks()

    while True:
        schedule.run_pending()
        time.sleep(1)