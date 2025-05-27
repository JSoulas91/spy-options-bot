# config.py

# -------------------------
# ALPACA PAPER TRADING CONFIGURATION
# -------------------------
ALPACA_API_KEY = 'your_alpaca_paper_api_key'
ALPACA_SECRET_KEY = 'your_alpaca_paper_secret_key'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# -------------------------
# TRADING SETTINGS
# -------------------------
DEFAULT_POSITION_SIZE = 0.1          # 10% of total equity per trade
MAX_DAY_TRADES = 3                   # Pattern Day Trading rule
AGGRESSIVE_TRADE_SIZE = 0.15         # 15% for aggressive mode
MIN_OPTION_EXPIRY_DAYS = 7           # Only trade contracts with at least 1 week to expiry

# -------------------------
# STRATEGY SETTINGS
# -------------------------
CONFIDENCE_THRESHOLD = 0.75          # Minimum confidence score to take a trade
STOP_LOSS_ATR_MULTIPLIER = 1.5       # ATR-based stop-loss
TRAILING_STOP_PERCENT = 0.10         # 10% trailing stop at profit
PREFERS_LIQUID_OPTIONS = True        # Filters options with high volume + tight spreads

# -------------------------
# TELEGRAM NOTIFICATIONS (Optional)
# -------------------------
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'your_telegram_chat_id'

# -------------------------
# TIME SETTINGS
# -------------------------
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
NO_NEW_TRADES_AFTER = "15:30"        # No new day trades after 3:30 PM

# -------------------------
# MISC
# -------------------------
USE_AGGRESSIVE_MODE = False
BLACKLISTED_DATES = []               # Dates to skip due to high-impact economic events
