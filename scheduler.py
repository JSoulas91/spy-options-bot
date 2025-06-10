# scheduler.py
import os
import time
import pytz
import schedule
import subprocess
from datetime import datetime

from utils.logger import bot_logger
from utils.logs import clean_old_logs, backup_logs
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import get_daily_trade_summary
from ml.retrain import retrain_model
from live_runner import run_market_open_tasks

# Health‑check
from monitor.health_check import update_status

# Online meta-agent update
from meta.online_meta_update import online_update

# ──────────────────────────────────────────────────────────────
# Global timezone handling: US/Eastern wall‑clock
os.environ["TZ"] = "US/Eastern"
if hasattr(time, "tzset"):          # Unix only
    time.tzset()

eastern  = pytz.timezone("US/Eastern")
telegram = TelegramNotifier()

# ─── Task helpers ──────────────────────────────────────────────
def send_daily_summary_task():
    try:
        summary_text = get_daily_trade_summary()
        telegram.send_message(f"📊 <b>Daily Performance Summary</b>\n\n{summary_text}")
        bot_logger.info("Daily summary sent via Telegram.")
    except Exception as e:
        bot_logger.error(f"Failed to send daily summary: {e}")

def retrain_model_task():
    try:
        bot_logger.info("Running model retraining task.")
        retrain_model()
    except Exception as e:
        bot_logger.error(f"Retraining task failed: {e}")

def clean_logs_task():
    try:
        bot_logger.info("Cleaning up old logs.")
        clean_old_logs()
    except Exception as e:
        bot_logger.error(f"Log cleanup failed: {e}")

def backup_logs_task():
    try:
        bot_logger.info("Backing up logs.")
        backup_logs()
    except Exception as e:
        bot_logger.error(f"Log backup failed: {e}")

def train_meta_agent_task():
    bot_logger.info("Starting daily PPO training …")
    try:
        result = subprocess.run(
            ["python", "meta/train_meta_agent.py"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )
        bot_logger.info("PPO training completed.")
        bot_logger.debug(f"PPO stdout:\n{result.stdout}")
        if result.stderr:
            bot_logger.warning(f"PPO stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired as e:
        bot_logger.error(f"PPO training timed out: {e}")
        telegram.send_message("⚠️ PPO training timed out after 5 minutes.")
    except Exception as e:
        bot_logger.error(f"PPO training failed: {e}")

def online_meta_update_task():
    try:
        bot_logger.info("Running online meta-agent update …")
        online_update()
        bot_logger.info("Online meta-agent update completed.")
    except Exception as e:
        bot_logger.error(f"Online meta-agent update failed: {e}")

def run_simulation_training_task():
    bot_logger.info("🏋️‍♂️ Running simulation training …")
    try:
        result = subprocess.run(
            ["python", "simulation/sim_train_full.py"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False
        )
        bot_logger.info("Simulation training completed.")
        bot_logger.debug(f"Simulation stdout:\n{result.stdout}")
        if result.stderr:
            bot_logger.warning(f"Simulation stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired as e:
        bot_logger.error(f"Simulation training timed out: {e}")
        telegram.send_message("⚠️ Simulation training timed out after 10 minutes.")
    except Exception as e:
        bot_logger.error(f"Simulation training failed: {e}")

def run_weekend_training_tasks():
    bot_logger.info("🏋️‍♂️ Starting weekend simulation + training sequence ...")
    telegram.send_message("🏋️‍♂️ Weekend simulation training started.")

    try:
        # 1️⃣ Run simulation training
        result = subprocess.run(
            ["python", "simulation/sim_train_full.py"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            check=False,
        )
        bot_logger.info("Simulation training completed.")
        if result.stderr:
            bot_logger.warning(f"Simulation training stderr:\n{result.stderr}")

        # 2️⃣ Run ML retrain (ml/retrain.py)
        bot_logger.info("Starting ML retraining after simulation …")
        retrain_model()
        bot_logger.info("ML retraining completed.")

        # 3️⃣ Run PPO training (meta/train_meta_agent.py)
        bot_logger.info("Starting PPO training after ML retrain …")
        result = subprocess.run(
            ["python", "meta/train_meta_agent.py"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            check=False,
        )
        bot_logger.info("PPO training completed.")
        if result.stderr:
            bot_logger.warning(f"PPO training stderr:\n{result.stderr}")

        telegram.send_message("✅ Weekend simulation and training completed successfully.")
    except Exception as e:
        bot_logger.error(f"Weekend training sequence failed: {e}")
        telegram.send_message(f"⚠️ Weekend training sequence failed: {e}")

# ─── Main scheduler loop ───────────────────────────────────────
def run_scheduler_loop():
    bot_logger.info("Scheduler started.")
    update_status("last_scheduler_start")  # ✅ heartbeat

    # Market‑hours automation
    schedule.every().day.at("09:30").do(run_market_open_tasks)

    # Post‑market automation (US/Eastern)
    schedule.every().day.at("16:45").do(send_daily_summary_task)
    schedule.every().day.at("17:00").do(retrain_model_task)
    schedule.every().day.at("17:20").do(clean_logs_task)
    schedule.every().day.at("17:30").do(backup_logs_task)
    schedule.every().day.at("17:40").do(online_meta_update_task)
    schedule.every().day.at("17:50").do(train_meta_agent_task)

    # 🔁 Weekend combined simulation + ML + PPO training
    schedule.every().saturday.at("12:00").do(run_weekend_training_tasks)
    schedule.every().sunday.at("12:00").do(run_weekend_training_tasks)

    # Hourly health‑check runner
    schedule.every().hour.at(":30").do(
        lambda: subprocess.run(["python", "monitor/run_monitor.py"])
    )

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            bot_logger.error(f"[Scheduler] Unexpected error: {e}")