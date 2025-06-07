# config.py
import os
from dotenv import load_dotenv
from utils.logger import bot_logger

load_dotenv()
bot_logger.info("🔧 Loading configuration from .env …")

def _env(name: str, default=None, required: bool = False):
    val = os.getenv(name, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required env var: {name}")
    return val

# ───────────────────────────────────────── PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ───────────────────────────────────────── TRADIER
TRADIER_API_TOKEN   = _env("TRADIER_API_TOKEN", required=True)
USE_TRADIER_SANDBOX = _env("USE_TRADIER_SANDBOX", "true").lower() == "true"
TRADIER_BASE_URL    = (
    "https://sandbox.tradier.com/v1"
    if USE_TRADIER_SANDBOX else
    "https://api.tradier.com/v1"
)

# ───────────────────────────────────────── TELEGRAM
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN", required=True)
TELEGRAM_CHAT_ID   = _env("TELEGRAM_CHAT_ID",   required=True)

# ───────────────────────────────────────── SIMULATION
SIMULATION_MODE          = _env("SIMULATION_MODE", "true").lower() == "true"
DEFAULT_SLIPPAGE_BPS     = float(_env("DEFAULT_SLIPPAGE_BPS",  3))      # 0.03 %
DEFAULT_SPREAD_BPS       = float(_env("DEFAULT_SPREAD_BPS",   4))       # 0.04 %
SIM_MIN_FILL_DELAY_MS    = int(_env("SIM_MIN_FILL_DELAY_MS", 50))
SIM_MAX_FILL_DELAY_MS    = int(_env("SIM_MAX_FILL_DELAY_MS",150))

# ───────────────────────────────────────── TRADING SETTINGS
DEFAULT_POSITION_SIZE = float(_env("DEFAULT_POSITION_SIZE", 0.10))      # 10 %
MAX_DAY_TRADES        = int(_env("MAX_DAY_TRADES", 3))
ENFORCE_PDT_LIMITS    = _env("ENFORCE_PDT_LIMITS", "false").lower() == "true"
MAX_OPEN_TRADES       = int(_env("MAX_OPEN_TRADES", 8))

# Dynamic sizing
ENABLE_DYNAMIC_SIZING = _env("ENABLE_DYNAMIC_SIZING", "true").lower() == "true"
MIN_POSITION_SIZE     = float(_env("MIN_POSITION_SIZE", 0.05))
MAX_POSITION_SIZE     = float(_env("MAX_POSITION_SIZE", 0.25))

# ───────────────────────────────────────── STRATEGY
CONFIDENCE_THRESHOLD     = float(_env("CONFIDENCE_THRESHOLD", 0.75))
STOP_LOSS_ATR_MULTIPLIER = float(_env("STOP_LOSS_ATR_MULTIPLIER", 1.5))
TRAILING_STOP_PERCENT    = float(_env("TRAILING_STOP_PERCENT", 0.10))

# ───────────────────────────────────────── VIX / VOL
VIX_MAX_THRESHOLD      = float(_env("VIX_MAX_THRESHOLD", 30.0))
VIX_MODERATE_THRESHOLD = float(_env("VIX_MODERATE_THRESHOLD", 25.0))
CONFIDENCE_STEP_UP     = float(_env("CONFIDENCE_STEP_UP", 0.05))

# ───────────────────────────────────────── META / PATHS
META_LOG_PATH  = os.path.join(BASE_DIR, "meta", "meta_log.jsonl")
META_INFO_PATH = os.path.join(BASE_DIR, "meta", "meta_agent_info.json")

bot_logger.info(f"✅ Config loaded – Simulation mode {'ON' if SIMULATION_MODE else 'OFF'}")