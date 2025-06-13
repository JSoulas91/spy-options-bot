import os
import time
import pytz
import schedule
import subprocess
import argparse
from utils.logger import bot_logger
from utils.log_cleanup import cleanup_logs_and_backups
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import get_daily_trade_summary
from ml.retrain import retrain_model
from live_runner import run_market_open_tasks
from monitor.health_check import update_status
from meta.online_meta_update import online_update

# ─── Timezone config (US/Eastern) ──────────────────────────────
os.environ["TZ"] = "US/Eastern"
if hasattr(time, "tzset"):
    time.tzset()
eastern = pytz.timezone("US/Eastern")
telegram = TelegramNotifier()

# ─── Task helpers ──────────────────────────────────────────────
def send_daily_summary_task():
    try:
        summary_text = get_daily_trade_summary()
        telegram.send_message(f"📊 <b>Daily Performance Summary</b>\n\n{summary_text}")
        bot_logger.info("✅ Daily summary sent.")
    except Exception as e:
        bot_logger.error(f"❌ Failed to send daily summary: {e}")
        telegram.send_message("⚠️ Error sending daily summary.")

def retrain_model_task():
    try:
        bot_logger.info("📈 Running ML model retraining …")
        retrain_model()
        bot_logger.info("✅ ML model retraining complete.")
    except Exception as e:
        bot_logger.error(f"❌ ML retraining failed: {e}")
        telegram.send_message("⚠️ ML retraining failed.")

def clean_logs_task():
    try:
        bot_logger.info("🧹 Cleaning old logs and backups …")
        cleanup_logs_and_backups()
    except Exception as e:
        bot_logger.error(f"❌ Log cleanup failed: {e}")

def train_meta_agent_task():
    try:
        bot_logger.info("🧠 Starting PPO meta-agent training …")
        result = subprocess.run(
            ["python", "meta/train_meta_agent.py"],
            capture_output=True, text=True, timeout=300
        )
        bot_logger.info("✅ PPO training finished.")
        if result.stdout: bot_logger.debug(f"PPO stdout:\n{result.stdout}")
        if result.stderr: bot_logger.warning(f"PPO stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        bot_logger.error("⏱️ PPO training timed out.")
        telegram.send_message("⚠️ PPO training timed out after 5 minutes.")
    except Exception as e:
        bot_logger.error(f"❌ PPO training error: {e}")
        telegram.send_message("⚠️ PPO training failed.")

def online_meta_update_task():
    try:
        bot_logger.info("🌐 Running online meta-agent update …")
        online_update()
        bot_logger.info("✅ Online update complete.")
    except Exception as e:
        bot_logger.error(f"❌ Online meta-agent update failed: {e}")
        telegram.send_message("⚠️ Online meta update failed.")

def run_simulation_training_task():
    try:
        bot_logger.info("🎮 Running full simulation training …")
        result = subprocess.run(
            ["python", "simulation/sim_train_full.py"],
            capture_output=True, text=True, timeout=600
        )
        bot_logger.info("✅ Simulation training done.")
        if result.stdout: bot_logger.debug(f"Sim stdout:\n{result.stdout}")
        if result.stderr: bot_logger.warning(f"Sim stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        bot_logger.error("⏱️ Simulation training timed out.")
        telegram.send_message("⚠️ Simulation training timed out (10 min).")
    except Exception as e:
        bot_logger.error(f"❌ Simulation training failed: {e}")
        telegram.send_message("⚠️ Simulation training failed.")

def run_weekend_training_tasks():
    bot_logger.info("🏋️ Starting weekend simulation + training …")
    telegram.send_message("🏋️ Weekend training started.")

    try:
        # Step 1: Simulation
        result = subprocess.run(
            ["python", "simulation/sim_train_full.py"],
            capture_output=True, text=True, timeout=1800
        )
        if result.stdout: bot_logger.debug(f"Sim stdout:\n{result.stdout}")
        if result.stderr: bot_logger.warning(f"Sim stderr:\n{result.stderr}")
        bot_logger.info("✅ Weekend simulation complete.")

        # Step 2: ML Retrain
        retrain_model()
        bot_logger.info("✅ ML retraining after simulation complete.")

        # Step 3: PPO Training
        result = subprocess.run(
            ["python", "meta/train_meta_agent.py"],
            capture_output=True, text=True, timeout=600
        )
        if result.stdout: bot_logger.debug(f"PPO stdout:\n{result.stdout}")
        if result.stderr: bot_logger.warning(f"PPO stderr:\n{result.stderr}")
        bot_logger.info("✅ PPO training after ML complete.")

        telegram.send_message("✅ Weekend simulation and training done.")
    except Exception as e:
        bot_logger.error(f"❌ Weekend training failure: {e}")
        telegram.send_message(f"⚠️ Weekend training failed:\n{e}")

# ─── Schedule setup ────────────────────────────────────────────
def run_scheduler_loop(debug=False):
    bot_logger.info("🕒 Scheduler started.")
    update_status("last_scheduler_start")

    # Weekday-only tasks
    schedule.every().monday.to.friday.at("09:30").do(run_market_open_tasks)
    schedule.every().monday.to.friday.at("16:45").do(send_daily_summary_task)
    schedule.every().monday.to.friday.at("17:00").do(retrain_model_task)
    schedule.every().monday.to.friday.at("17:20").do(clean_logs_task)
    schedule.every().monday.to.friday.at("17:40").do(online_meta_update_task)
    schedule.every().monday.to.friday.at("17:50").do(train_meta_agent_task)

    # Weekend simulation + training
    schedule.every().saturday.at("12:00").do(run_weekend_training_tasks)
    schedule.every().sunday.at("12:00").do(run_weekend_training_tasks)

    # Hourly monitor pings
    schedule.every().hour.at(":30").do(
        lambda: subprocess.run(["python", "monitor/run_monitor.py"])
    )

    # Main loop
    while True:
        try:
            schedule.run_pending()
            if debug:
                bot_logger.debug("⏳ Waiting for next scheduled task …")
            time.sleep(1)
        except Exception as e:
            bot_logger.error(f"❌ Scheduler error: {e}")

# ─── CLI entrypoint ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY Options Bot Scheduler")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    if args.debug:
        bot_logger.setLevel("DEBUG")
        bot_logger.debug("🔍 Debug mode enabled.")

    run_scheduler_loop(debug=args.debug)