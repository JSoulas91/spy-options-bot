import os
from dotenv import load_dotenv
from utils.logger import bot_logger

# Load environment variables from .env file
load_dotenv()
bot_logger.info("🔧 Loading configuration from .env...")

def get_env_var(name, required=True, default=None):
    value = os.getenv(name, default)
    if required and value is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value

# === ALPACA PAPER TRADING CONFIGURATION ===
ALPACA_API_KEY = get_env_var('ALPACA_API_KEY')
ALPACA_SECRET_KEY = get_env_var('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = get_env_var('ALPACA_BASE_URL')

# === TELEGRAM NOTIFICATIONS ===
TELEGRAM_BOT_TOKEN = get_env_var('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = get_env_var('TELEGRAM_CHAT_ID')

# === TRADING SETTINGS ===
DEFAULT_POSITION_SIZE = float(os.getenv('DEFAULT_POSITION_SIZE', 0.1))
MAX_DAY_TRADES = int(os.getenv('MAX_DAY_TRADES', 3))
AGGRESSIVE_TRADE_SIZE = float(os.getenv('AGGRESSIVE_TRADE_SIZE', 0.15))
MIN_OPTION_EXPIRY_DAYS = int(os.getenv('MIN_OPTION_EXPIRY_DAYS', 7))

# === STRATEGY SETTINGS ===
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.75))
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv('STOP_LOSS_ATR_MULTIPLIER', 1.5))
TRAILING_STOP_PERCENT = float(os.getenv('TRAILING_STOP_PERCENT', 0.10))
PREFERS_LIQUID_OPTIONS = os.getenv('PREFERS_LIQUID_OPTIONS', 'true').lower() == 'true'

# === TIME SETTINGS ===
MARKET_OPEN = os.getenv('MARKET_OPEN', "09:30")
MARKET_CLOSE = os.getenv('MARKET_CLOSE', "16:00")
NO_NEW_TRADES_AFTER = os.getenv('NO_NEW_TRADES_AFTER', "15:30")

# === MISC ===
USE_AGGRESSIVE_MODE = os.getenv('USE_AGGRESSIVE_MODE', 'false').lower() == 'true'
BLACKLISTED_DATES = []  # You can load this from a file or API if needed

bot_logger.info("✅ Configuration loaded successfully.")