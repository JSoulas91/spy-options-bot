import schedule
import time
from data.data_collector import DataCollector
from strategy.strategy import TradingStrategy
from trading.trade_manager import TradeManager
from reporting.telegram_bot import TelegramBot
from reporting.performance_dashboard import PerformanceDashboard
from ml.retrain import retrain_model  # Make sure this path is correct

# Initialize components
data_collector = DataCollector()
strategy = TradingStrategy()
trade_manager = TradeManager()
telegram_bot = TelegramBot()
dashboard = PerformanceDashboard()

def run_bot():
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
        telegram_bot.send_message(f"❌ Error occurred: {str(e)}")

def send_daily_summary():
    summary = dashboard.generate_summary()
    telegram_bot.send_message(summary)

def run_retraining():
    try:
        retrain_model()
        telegram_bot.send_message("🧠 ML model retrained successfully.")
    except Exception as e:
        telegram_bot.send_message(f"❌ Retrain error: {str(e)}")

# Schedule tasks
schedule.every().day.at("09:30").do(run_bot)             # Market open trade
schedule.every().day.at("16:45").do(send_daily_summary)  # Summary 15 min after market close
schedule.every().day.at("17:00").do(run_retraining)      # Retraining after close

if __name__ == "__main__":
    telegram_bot.send_message("🚀 SPY Options Bot Started")
    while True:
        schedule.run_pending()
        time.sleep(1)