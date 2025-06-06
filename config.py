# config.py
import os
from dotenv import load_dotenv
from utils.logger import bot_logger

load_dotenv()
bot_logger.info("🔧 Loading configuration from .env…")

def get_env_var(name, required=True, default=None):
    val = os.getenv(name, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return val

# ─────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────
# TRADIER
# ─────────────────────────────────────────────────────
TRADIER_API_TOKEN   = get_env_var("TRADIER_API_TOKEN")
USE_TRADIER_SANDBOX = os.getenv("USE_TRADIER_SANDBOX", "true").lower() == "true"
TRADIER_BASE_URL    = "https://sandbox.tradier.com/v1" if USE_TRADIER_SANDBOX else "https://api.tradier.com/v1"

# ─────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = get_env_var("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = get_env_var("TELEGRAM_CHAT_ID")

# ─────────────────────────────────────────────────────
# TRADING SETTINGS
# ─────────────────────────────────────────────────────
DEFAULT_POSITION_SIZE = float(os.getenv("DEFAULT_POSITION_SIZE", 0.10))   # 10% of account by default
MAX_DAY_TRADES        = int(os.getenv("MAX_DAY_TRADES", 3))
ENFORCE_PDT_LIMITS    = os.getenv("ENFORCE_PDT_LIMITS", "false").lower() == "true"
MAX_OPEN_TRADES       = int(os.getenv("MAX_OPEN_TRADES", 8))

# → NEW dynamic sizing knobs
ENABLE_DYNAMIC_SIZING = os.getenv("ENABLE_DYNAMIC_SIZING", "true").lower() == "true"
MIN_POSITION_SIZE     = float(os.getenv("MIN_POSITION_SIZE", 0.05))   # 5 %
MAX_POSITION_SIZE     = float(os.getenv("MAX_POSITION_SIZE", 0.25))   # 25 %

# ─────────────────────────────────────────────────────
# STRATEGY
# ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD     = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", 1.5))
TRAILING_STOP_PERCENT    = float(os.getenv("TRAILING_STOP_PERCENT", 0.10))
PREFERS_LIQUID_OPTIONS   = os.getenv("PREFERS_LIQUID_OPTIONS", "true").lower() == "true"

# ─────────────────────────────────────────────────────
# TIME SETTINGS
# ─────────────────────────────────────────────────────
MARKET_OPEN        = os.getenv("MARKET_OPEN", "09:30")
MARKET_CLOSE       = os.getenv("MARKET_CLOSE", "16:00")
NO_NEW_TRADES_AFTER= os.getenv("NO_NEW_TRADES_AFTER", "15:30")

# ─────────────────────────────────────────────────────
# VIX
# ─────────────────────────────────────────────────────
VIX_MAX_THRESHOLD      = float(os.getenv("VIX_MAX_THRESHOLD", 30.0))
VIX_MODERATE_THRESHOLD = float(os.getenv("VIX_MODERATE_THRESHOLD", 25.0))
VIX_SAFE_FOR_SWING     = float(os.getenv("VIX_SAFE_FOR_SWING", 20.0))
CONFIDENCE_STEP_UP     = float(os.getenv("CONFIDENCE_STEP_UP", 0.05))

# ─────────────────────────────────────────────────────
# META‑AGENT
# ─────────────────────────────────────────────────────
META_LOG_PATH = os.path.join(BASE_DIR, "meta", "meta_log.jsonl")
META_INFO_PATH= os.path.join(BASE_DIR, "meta", "meta_agent_info.json")

bot_logger.info("✅ Configuration loaded successfully.")