import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === ALPACA PAPER TRADING CONFIGURATION ===
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

# === TELEGRAM NOTIFICATIONS ===
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# === TRADING SETTINGS ===
DEFAULT_POSITION_SIZE = 0.1          # 10% of total equity per trade
MAX_DAY_TRADES = 3                   # Pattern Day Trading rule
AGGRESSIVE_TRADE_SIZE = 0.15         # 15% for aggressive mode
MIN_OPTION_EXPIRY_DAYS = 7           # Only trade contracts with at least 1 week to expiry

# === STRATEGY SETTINGS ===
CONFIDENCE_THRESHOLD = 0.75          # Minimum confidence score to take a trade
STOP_LOSS_ATR_MULTIPLIER = 1.5       # ATR-based stop-loss
TRAILING_STOP_PERCENT = 0.10         # 10% trailing stop at profit
PREFERS_LIQUID_OPTIONS = True        # Filters options with high volume + tight spreads

# === TIME SETTINGS ===
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
NO_NEW_TRADES_AFTER = "15:30"        # No new day trades after 3:30 PM

# === MISC ===
USE_AGGRESSIVE_MODE = False
BLACKLISTED_DATES = []               # Dates to skip due to high-impact economic events