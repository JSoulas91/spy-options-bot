# config.py

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

# === BASE PATH ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
ENFORCE_PDT_LIMITS = os.getenv('ENFORCE_PDT_LIMITS', 'false').lower() == 'true'
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

# === ADVANCED STRATEGY FILTERS ===
ENABLE_EVENT_BLACKOUT = os.getenv("ENABLE_EVENT_BLACKOUT", "true").lower() == "true"
ENABLE_VIX_THROTTLING = os.getenv("ENABLE_VIX_THROTTLING", "true").lower() == "true"
ENABLE_FED_SPEAKER_FILTER = os.getenv("ENABLE_FED_SPEAKER_FILTER", "true").lower() == "true"
ENABLE_ADAPTIVE_CONFIDENCE = os.getenv("ENABLE_ADAPTIVE_CONFIDENCE", "true").lower() == "true"

# === VIX SETTINGS ===
VIX_MAX_THRESHOLD = float(os.getenv("VIX_MAX_THRESHOLD", 30.0))
VIX_MODERATE_THRESHOLD = float(os.getenv("VIX_MODERATE_THRESHOLD", 25.0))
VIX_SAFE_FOR_SWING = float(os.getenv("VIX_SAFE_FOR_SWING", 20.0))

# === CONFIDENCE ADJUSTMENTS ===
CONFIDENCE_STEP_UP = float(os.getenv("CONFIDENCE_STEP_UP", 0.05))

# === ADAPTIVE CONFIDENCE THRESHOLDS ===
BASE_CONFIDENCE_THRESHOLD = float(os.getenv("BASE_CONFIDENCE_THRESHOLD", 0.55))
OPENING_RANGE_THRESHOLD = float(os.getenv("OPENING_RANGE_THRESHOLD", 0.50))
HIGH_VIX_THRESHOLD = float(os.getenv("HIGH_VIX_THRESHOLD", 0.65))
SWING_TRADE_CONFIDENCE_THRESHOLD = float(os.getenv("SWING_TRADE_CONFIDENCE_THRESHOLD", 0.70))

# === RETRY SETTINGS ===
MAX_RETRIES_PER_TRADE = int(os.getenv("MAX_RETRIES_PER_TRADE", 3))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", 5))

# === OPTION FILTER SETTINGS ===
OPTION_TYPE_FILTER = os.getenv("OPTION_TYPE_FILTER", "both").lower()
MIN_EXPIRY_DAYS = int(os.getenv("MIN_OPTION_EXPIRY_DAYS", 7))

MIN_DELTA = float(os.getenv("MIN_DELTA", 0.25))
MAX_DELTA = float(os.getenv("MAX_DELTA", 0.75))
MAX_THETA = float(os.getenv("MAX_THETA", -0.01))
MAX_VEGA = float(os.getenv("MAX_VEGA", 0.15))
MAX_GAMMA = float(os.getenv("MAX_GAMMA", 0.15))

MIN_VOLUME = int(os.getenv("MIN_VOLUME", 100))
MIN_OPEN_INTEREST = int(os.getenv("MIN_OPEN_INTEREST", 100))

MAX_BID_ASK_SPREAD_PCT = float(os.getenv("MAX_BID_ASK_SPREAD_PCT", 0.15))
MIN_MONEYNESS = float(os.getenv("MIN_MONEYNESS", 0.90))
MAX_MONEYNESS = float(os.getenv("MAX_MONEYNESS", 1.10))
ENABLE_DIRECTIONAL_SKEW = os.getenv("ENABLE_DIRECTIONAL_SKEW", "true").lower() == "true"

DEBUG_OPTION_FILTER = os.getenv("DEBUG_OPTION_FILTER", "false").lower() == "true"

# === META-AGENT SETTINGS ===
META_STATE_LOOKBACK_MINUTES = int(os.getenv("META_STATE_LOOKBACK_MINUTES", 720))
META_AGENT_ENABLED = os.getenv("META_AGENT_ENABLED", "true").lower() == "true"
META_LOG_PATH = os.path.join(BASE_DIR, "meta", "meta_log.jsonl")
META_INFO_PATH = os.path.join(BASE_DIR, "meta", "meta_agent_info.json")  # <-- ✅ Added

bot_logger.info("✅ Configuration loaded successfully.")