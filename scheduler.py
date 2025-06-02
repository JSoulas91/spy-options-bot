# scheduler.py

import schedule
import time
import pytz
from datetime import datetime
import subprocess

from utils.logger import bot_logger
from utils.logs import clean_old_logs, backup_logs
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import get_daily_trade_summary
from retrain import retrain_model
from main import run_market_open_tasks

eastern = pytz.timezone('US/Eastern')
telegram = TelegramNotifier()

def send_daily_summary_task():
    try:
        summary_text = get_daily_trade_summary()
        telegram.send_message(f"📊 <b>Daily Performance Summary</b>\n\n{summary_text}")
        bot_logger.info("Daily summary sent via Telegram.")
    except Exception as e:
        bot_logger.error(f"Failed to send daily summary: {str(e)}")

def retrain_model_task():
    try:
        bot_logger.info("Running model retraining task.")
        retrain_model()
    except Exception as e:
        bot_logger.error(f"Retraining task failed: {str(e)}")

def clean_logs_task():
    try:
        bot_logger.info("Cleaning up old logs.")
        clean_old_logs()
    except Exception as e:
        bot_logger.error(f"Log cleanup failed: {str(e)}")

def backup_logs_task():
    try:
        bot_logger.info("Backing up logs.")
        backup_logs()
    except Exception as e:
        bot_logger.error(f"Log backup failed: {str(e)}")

def train_meta_agent_task():
    bot_logger.info("Starting daily PPO training...")
    try:
        result = subprocess.run(["python", "meta/train_meta_agent.py"], capture_output=True, text=True)
        bot_logger.info("PPO training completed successfully.")
        bot_logger.debug(f"PPO Output: {result.stdout}")
        if result.stderr:
            bot_logger.warning(f"PPO stderr: {result.stderr}")
    except Exception as e:
        bot_logger.error(f"PPO training failed: {str(e)}")

def run_scheduler_loop():
    bot_logger.info("Scheduler started.")

    # Scheduling tasks
    schedule.every().day.at("09:30").do(run_market_open_tasks)
    schedule.every().day.at("16:45").do(send_daily_summary_task)
    schedule.every().day.at("17:00").do(retrain_model_task)
    schedule.every().day.at("17:20").do(clean_logs_task)
    schedule.every().day.at("17:30").do(backup_logs_task)
    schedule.every().day.at("17:50").do(train_meta_agent_task)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            bot_logger.error(f"[Scheduler] Unexpected error in main loop: {str(e)}")