import schedule
import time
import pytz
from datetime import datetime
import subprocess
from utils.logger import bot_logger
from retrain import retrain_model
from main import run_bot
from utils.logs import clean_old_logs, backup_logs

# Time zone setup
eastern = pytz.timezone('US/Eastern')

def run_trading_bot_task():
    bot_logger.info("Running trading bot task.")
    run_bot()

def retrain_model_task():
    bot_logger.info("Running model retraining task.")
    retrain_model()

def send_daily_summary_task():
    bot_logger.info("Sending daily performance summary.")
    # Placeholder: add actual summary logic if needed

def clean_logs_task():
    bot_logger.info("Cleaning up old logs.")
    clean_old_logs()

def backup_logs_task():
    bot_logger.info("Backing up logs.")
    backup_logs()

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

# Scheduling tasks
schedule.every().day.at("09:30").do(run_trading_bot_task)        # Market open
schedule.every().day.at("16:45").do(send_daily_summary_task)     # Summary after market close
schedule.every().day.at("17:00").do(retrain_model_task)          # Retrain ML model
schedule.every().day.at("17:20").do(clean_logs_task)             # Clean up logs
schedule.every().day.at("17:30").do(backup_logs_task)            # Backup logs
schedule.every().day.at("17:50").do(train_meta_agent_task)       # Train PPO meta-agent

# Main loop
if __name__ == "__main__":
    bot_logger.info("Scheduler started.")
    while True:
        schedule.run_pending()
        time.sleep(1)