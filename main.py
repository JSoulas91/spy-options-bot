import schedule
import time
from datetime import datetime
import pytz
import os
import subprocess

from config import (
    MARKET_OPEN,
    MARKET_CLOSE,
    NO_NEW_TRADES_AFTER
)

from data.data_collector import DataCollector
from strategy.strategy import TradingStrategy
from trading.trade_manager import TradeManager
from reporting.telegram_bot import TelegramBot
from reporting.performance_dashboard import PerformanceDashboard
from utils.log_cleanup import cleanup_logs
from utils.backup import backup_data

# Timezone
eastern = pytz.timezone('US/Eastern')

# Initialize components
data_collector = DataCollector()
strategy = TradingStrategy()
trade_manager = TradeManager()
telegram_bot = TelegramBot()
dashboard = PerformanceDashboard()

def run_bot():
    now = datetime.now(eastern)
    if now.strftime('%H:%M') == MARKET_OPEN:
        try:
            market_data = data_collector.collect_market_data()
            signal, confidence = strategy.generate_trade_signal(market_data)

            if signal:
                option_contract = trade_manager.select_option_contract(signal, market_data)
                trade_result = trade_manager.execute_trade(option_contract, signal, confidence)
                dashboard.record_trade(trade_result)

                msg = (
                    f"📥 Trade Executed\n"
                    f"🧠 Signal: {signal.upper()}\n"
                    f"📈 Confidence: {confidence:.2f}\n"
                    f"🎯 Result: {trade_result['pnl']}%\n"
                )
                telegram_bot.send_message(msg)
            else:
                telegram_bot.send_message("🤖 No trade signal generated.")

        except Exception as e:
            telegram_bot.send_message(f"❌ Error occurred during trade: {str(e)}")

def send_daily_summary():
    now = datetime.now(eastern)
    if now.strftime('%H:%M') == "16:45":
        try:
            summary = dashboard.generate_summary()
            telegram_bot.send_message(summary)
        except Exception as e:
            telegram_bot.send_message(f"❌ Error during summary: {str(e)}")

def retrain_model_and_maintenance():
    now = datetime.now(eastern)
    if now.strftime('%H:%M') == "16:50":
        try:
            telegram_bot.send_message("🔁 Starting daily retraining and cleanup...")
            # Run retraining script
            subprocess.run(["python3", "ml/retrain.py"], check=True)
            # Clean up logs
            cleanup_logs()
            # Backup data
            backup_data()
            telegram_bot.send_message("✅ Daily retraining, cleanup, and backup complete.")
        except Exception as e:
            telegram_bot.send_message(f"❌ Error during retraining or maintenance: {str(e)}")

if __name__ == "__main__":
    telegram_bot.send_message("🚀 SPY Options Bot Started")

    # Schedule to run checks every minute
    schedule.every().minute.do(run_bot)
    schedule.every().minute.do(send_daily_summary)
    schedule.every().minute.do(retrain_model_and_maintenance)

    while True:
        schedule.run_pending()
        time.sleep(1)