import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === API Keys ===
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY")
TRADIER_ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID")
TRADIER_BASE_URL = "https://api.tradier.com/v1"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === General Bot Settings ===
UNDERLYING_SYMBOL = "SPY"
MAX_DAY_TRADES = 3
DAY_TRADE_END_BUFFER_MINUTES = 30  # Stop new day trades 30 minutes before close
SWING_TRADE_LOOKBACK_DAYS = 90

# === Option Filters ===
MIN_DTE_DAYS = 7
DELTA_RANGE = (0.2, 0.5)
VEGA_MAX = 0.5
GAMMA_MAX = 0.5
MIN_VOLUME = 100
MIN_OPEN_INTEREST = 100

# === Risk Management ===
BASE_POSITION_SIZE = 0.10  # 10% of portfolio unless aggressive mode
AGGRESSIVE_POSITION_SIZE = 0.15
MAX_OPEN_POSITIONS = 3
USE_DYNAMIC_SIZING = True

# === Stop-Loss & Take-Profit ===
USE_ATR_STOP = True
TRAILING_STOP_AT_PROFIT = 0.10  # 10% trailing stop if profit hits

# === Confidence Filter ===
CONFIDENCE_THRESHOLD = 0.7  # Must be >= to enter trade

# === Economic Event Handling ===
ENABLE_EVENT_FILTER = True
ECONOMIC_EVENTS_URL = "https://api.tradingeconomics.com/calendar"

# === Logging ===
ENABLE_TELEGRAM_LOGGING = True
