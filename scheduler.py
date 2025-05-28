# scheduler.py

import schedule
import time
from datetime import datetime
from pytz import timezone
from trade_manager import manage_open_positions, get_open_positions
from retrain import retrain_model  # ✅ Updated import
from telegram_bot import TelegramBot
from utils.backup import backup_logs
from utils.log_cleanup import cleanup_logs
from strategy import run_trading_strategy
from config import MARKET_OPEN, MARKET_CLOSE

# Timezone settings
eastern = timezone("US/Eastern")
bot = TelegramBot()

def run_market_open():
    print(f"[{datetime.now()}] Running trading strategy...")
    run_trading_strategy()

def run_end_of_day_tasks():
    print(f"[{datetime.now()}] Running EOD tasks: summary, retrain, cleanup...")
    
    # Manage open positions
    positions = get_open_positions()
    manage_open_positions(positions)

    # Retrain model
    retrain_model()  # ✅ Updated function call

    # Backup logs
    backup_logs()

    # Cleanup logs
    cleanup_logs()

    # Send summary report (dummy trades passed here, replace with real results)
    trades_today = []  # Replace with actual trades if tracked
    bot.send_daily_summary(trades_today)

def start_scheduler():
    # Schedule trading strategy at market open
    schedule.every().day.at(MARKET_OPEN).do(run_market_open)

    # Schedule end-of-day tasks (summary, backup, cleanup, retrain)
    schedule.every().day.at("16:45").do(run_end_of_day_tasks)

    print("Scheduler is running...")

    while True:
        schedule.run_pending()
        time.sleep(1)