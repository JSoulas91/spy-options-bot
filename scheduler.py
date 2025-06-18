import os
import time
import pytz
import argparse
import schedule
import subprocess
from utils.logger import bot_logger
from utils.log_cleanup import cleanup_logs_and_backups
from utils.telegram_notifier import TelegramNotifier
from utils.trade_logger import get_daily_trade_summary
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

def clean_logs_task():
    try:
        bot_logger.info("🧹 Cleaning old logs and backups …")
        cleanup_logs_and_backups()
    except Exception as e:
        bot_logger.error(f"❌ Log cleanup failed: {e}")

def online_meta_update_task():
    try:
        bot_logger.info("🌐 Running online meta-agent update …")
        online_update()
        bot_logger.info("✅ Online update complete.")
    except Exception as e:
        bot_logger.error(f"❌ Online meta-agent update failed: {e}")
        telegram.send_message("⚠️ Online meta update failed.")

# ─── Schedule setup ────────────────────────────────────────────
def run_scheduler_loop(debug=False):
    if debug:
        bot_logger.debug("🔍 Debug mode enabled.")

    bot_logger.info("🕒 Scheduler started.")
    update_status("last_scheduler_start")

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    for day in weekdays:
        getattr(schedule.every(), day).at("09:30").do(run_market_open_tasks)
        getattr(schedule.every(), day).at("16:45").do(send_daily_summary_task)
        getattr(schedule.every(), day).at("17:20").do(clean_logs_task)
        getattr(schedule.every(), day).at("17:40").do(online_meta_update_task)

    # Hourly monitor pings
    schedule.every().hour.at(":30").do(
        lambda: subprocess.run(["python", "monitor/run_monitor.py"])
    )

    # Loop execution
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            bot_logger.error(f"❌ Scheduler error: {e}")

# ─── Entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode logging.")
    args = parser.parse_args()
    run_scheduler_loop(debug=args.debug)