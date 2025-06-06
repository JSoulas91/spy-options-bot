# config.py
import os
from dotenv import load_dotenv
from utils.logger import bot_logger

load_dotenv()
bot_logger.info("🔧 Loading configuration from .env...")

def get_env_var(name, required=True, default=None):
    val = os.getenv(name, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return val

# === BASE PATH ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === TRADIER API ===
# Accept either TRADIER_API_KEY or TRADIER_API_TOKEN for backward‑compat
TRADIER_API_KEY   = os.getenv("TRADIER_API_KEY")   or os.getenv("TRADIER_API_TOKEN")
if TRADIER_API_KEY is None:
    raise EnvironmentError("Missing Tradier key—set TRADIER_API_KEY in .env")

USE_TRADIER_SANDBOX = os.getenv("USE_TRADIER_SANDBOX", "true").lower() == "true"
TRADIER_BASE_URL = (
    "https://sandbox.tradier.com/v1" if USE_TRADIER_SANDBOX
    else "https://api.tradier.com/v1"
)

# Make sure fetcher modules see the URL/key even if they don't import config
os.environ.setdefault("TRADIER_BASE_URL", TRADIER_BASE_URL)
os.environ.setdefault("TRADIER_API_KEY",  TRADIER_API_KEY)

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN = get_env_var("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = get_env_var("TELEGRAM_CHAT_ID")

# === TRADING LIMITS ===
DEFAULT_POSITION_SIZE = float(os.getenv("DEFAULT_POSITION_SIZE", 0.1))
MAX_DAY_TRADES        = int(os.getenv("MAX_DAY_TRADES", 3))
MAX_OPEN_TRADES       = int(os.getenv("MAX_OPEN_TRADES", 8))
ENFORCE_PDT_LIMITS    = os.getenv("ENFORCE_PDT_LIMITS", "false").lower() == "true"
AGGRESSIVE_TRADE_SIZE = float(os.getenv("AGGRESSIVE_TRADE_SIZE", 0.15))
MIN_OPTION_EXPIRY_DAYS= int(os.getenv("MIN_OPTION_EXPIRY_DAYS", 7))

# === STRATEGY PARAMETERS (unchanged) ===
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", 1.5))
TRAILING_STOP_PERCENT    = float(os.getenv("TRAILING_STOP_PERCENT", 0.10))
PREFERS_LIQUID_OPTIONS   = os.getenv("PREFERS_LIQUID_OPTIONS", "true").lower() == "true"

# ... (rest of your config remains identical) ...

bot_logger.info(
    f"✅ Config loaded. "
    f"Tradier {'SANDBOX' if USE_TRADIER_SANDBOX else 'LIVE'} endpoint={TRADIER_BASE_URL}"
)