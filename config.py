# config.py
"""
Centralised configuration loader.

• Reads .env via python‑dotenv
• Exposes strongly‑typed constants with sane defaults
• Now includes simulation + dynamic‑sizing knobs
"""

import os
from dotenv import load_dotenv
from utils.logger import bot_logger

load_dotenv()
bot_logger.info("🔧 Loading configuration from .env…")

# ────────────────────────────────────────────────────────────
def get_env_var(name: str, required: bool = True, default=None):
    val = os.getenv(name, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required env var: {name}")
    return val

# ──────────────────────────────────────────────────────────── PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────── TRADIER
TRADIER_API_TOKEN   = get_env_var("TRADIER_API_TOKEN")
USE_TRADIER_SANDBOX = os.getenv("USE_TRADIER_SANDBOX", "true").lower() == "true"
TRADIER_BASE_URL    = (
    "https://sandbox.tradier.com/v1"
    if USE_TRADIER_SANDBOX else
    "https://api.tradier.com/v1"
)

# ──────────────────────────────────────────────────────────── TELEGRAM
TELEGRAM_BOT_TOKEN = get_env_var("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = get_env_var("TELEGRAM_CHAT_ID")

# ──────────────────────────────────────────────────────────── TRADING SETTINGS
DEFAULT_POSITION_SIZE = float(os.getenv("DEFAULT_POSITION_SIZE", 0.10))  # 10 %
MAX_OPEN_TRADES       = int(os.getenv("MAX_OPEN_TRADES", 8))
MAX_DAY_TRADES        = int(os.getenv("MAX_DAY_TRADES", 3))
ENFORCE_PDT_LIMITS    = os.getenv("ENFORCE_PDT_LIMITS", "false").lower() == "true"

# —— dynamic sizing (entry.py uses these)
ENABLE_DYNAMIC_SIZING = os.getenv("ENABLE_DYNAMIC_SIZING", "true").lower() == "true"
MIN_POSITION_SIZE     = float(os.getenv("MIN_POSITION_SIZE", 0.05))  # 5 %
MAX_POSITION_SIZE     = float(os.getenv("MAX_POSITION_SIZE", 0.25))  # 25 %

# ──────────────────────────────────────────────────────────── STRATEGY
CONFIDENCE_THRESHOLD     = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", 1.5))
TRAILING_STOP_PERCENT    = float(os.getenv("TRAILING_STOP_PERCENT", 0.10))
PREFERS_LIQUID_OPTIONS   = os.getenv("PREFERS_LIQUID_OPTIONS", "true").lower() == "true"

# ──────────────────────────────────────────────────────────── TIME
MARKET_OPEN         = os.getenv("MARKET_OPEN", "09:30")
MARKET_CLOSE        = os.getenv("MARKET_CLOSE", "16:00")
NO_NEW_TRADES_AFTER = os.getenv("NO_NEW_TRADES_AFTER", "15:30")

# ──────────────────────────────────────────────────────────── VOLATILITY / VIX
VIX_MAX_THRESHOLD      = float(os.getenv("VIX_MAX_THRESHOLD", 30.0))
VIX_MODERATE_THRESHOLD = float(os.getenv("VIX_MODERATE_THRESHOLD", 25.0))
VIX_SAFE_FOR_SWING     = float(os.getenv("VIX_SAFE_FOR_SWING", 20.0))
CONFIDENCE_STEP_UP     = float(os.getenv("CONFIDENCE_STEP_UP", 0.05))

# ──────────────────────────────────────────────────────────── SIMULATION MODE
SIMULATION_MODE       = os.getenv("SIMULATION_MODE", "false").lower() == "true"
DEFAULT_SLIPPAGE_BPS  = float(os.getenv("DEFAULT_SLIPPAGE_BPS", 5))    # 5 bps = 0.05 %
DEFAULT_SPREAD_BPS    = float(os.getenv("DEFAULT_SPREAD_BPS", 8))     # bid‑ask half‑spread model
SIM_MIN_FILL_DELAY_MS = int(os.getenv("SIM_MIN_FILL_DELAY_MS", 50))
SIM_MAX_FILL_DELAY_MS = int(os.getenv("SIM_MAX_FILL_DELAY_MS", 150))

# ──────────────────────────────────────────────────────────── META‑AGENT PATHS
META_LOG_PATH  = os.path.join(BASE_DIR, "meta", "meta_log.jsonl")
META_INFO_PATH = os.path.join(BASE_DIR, "meta", "meta_agent_info.json")

bot_logger.info("✅ Configuration loaded.")