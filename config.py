import os
import datetime
from dotenv import load_dotenv

load_dotenv()

# === TRADIER KEYS AND URLS ===
TRADIER_API_TOKEN    = os.getenv("TRADIER_API_TOKEN", "")
TRADIER_ACCOUNT_ID   = os.getenv("TRADIER_ACCOUNT_ID", "")
USE_LIVE_TRADIER     = os.getenv("USE_LIVE_TRADIER", "false").lower() == "true"

if USE_LIVE_TRADIER:
    TRADIER_BASE_URL = "https://api.tradier.com/v1"
else:
    TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")

# === TRADE SETTINGS ===
DEFAULT_POSITION_SIZE = float(os.getenv("DEFAULT_POSITION_SIZE", 0.10))
MAX_OPEN_TRADES       = int(os.getenv("MAX_OPEN_TRADES", 8))
MAX_DAY_TRADES        = int(os.getenv("MAX_DAY_TRADES", 100))
ENFORCE_PDT_LIMITS    = os.getenv("ENFORCE_PDT_LIMITS", "false").lower() == "true"
AGGRESSIVE_TRADE_SIZE = float(os.getenv("AGGRESSIVE_TRADE_SIZE", 0.15))
MIN_OPTION_EXPIRY_DAYS = int(os.getenv("MIN_OPTION_EXPIRY_DAYS", 7))

# === STRATEGY ===
CONFIDENCE_THRESHOLD      = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
STOP_LOSS_ATR_MULTIPLIER  = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", 1.5))
TRAILING_STOP_PERCENT     = float(os.getenv("TRAILING_STOP_PERCENT", 0.10))
PREFERS_LIQUID_OPTIONS    = os.getenv("PREFERS_LIQUID_OPTIONS", "true").lower() == "true"
MIN_META_CONFIDENCE = float(os.getenv("MIN_META_CONFIDENCE", 0.6))

# === TIME ===
MARKET_OPEN         = os.getenv("MARKET_OPEN", "09:30")
MARKET_CLOSE        = os.getenv("MARKET_CLOSE", "16:00")
NO_NEW_TRADES_AFTER = os.getenv("NO_NEW_TRADES_AFTER", "15:30")

# === FLAGS ===
USE_AGGRESSIVE_MODE       = os.getenv("USE_AGGRESSIVE_MODE", "false").lower() == "true"
ENABLE_EVENT_BLACKOUT     = os.getenv("ENABLE_EVENT_BLACKOUT", "true").lower() == "true"
ENABLE_VIX_THROTTLING     = os.getenv("ENABLE_VIX_THROTTLING", "true").lower() == "true"
ENABLE_FED_SPEAKER_FILTER = os.getenv("ENABLE_FED_SPEAKER_FILTER", "true").lower() == "true"
ENABLE_ADAPTIVE_CONFIDENCE = os.getenv("ENABLE_ADAPTIVE_CONFIDENCE", "true").lower() == "true"

# === VIX SETTINGS ===
VIX_MAX_THRESHOLD      = float(os.getenv("VIX_MAX_THRESHOLD", 30.0))
VIX_MODERATE_THRESHOLD = float(os.getenv("VIX_MODERATE_THRESHOLD", 25.0))
VIX_SAFE_FOR_SWING     = float(os.getenv("VIX_SAFE_FOR_SWING", 20.0))
CONFIDENCE_STEP_UP     = float(os.getenv("CONFIDENCE_STEP_UP", 0.05))

# === CONFIDENCE THRESHOLDS ===
BASE_CONFIDENCE_THRESHOLD          = float(os.getenv("BASE_CONFIDENCE_THRESHOLD", 0.55))
OPENING_RANGE_THRESHOLD            = float(os.getenv("OPENING_RANGE_THRESHOLD", 0.50))
HIGH_VIX_THRESHOLD                 = float(os.getenv("HIGH_VIX_THRESHOLD", 0.65))
SWING_TRADE_CONFIDENCE_THRESHOLD   = float(os.getenv("SWING_TRADE_CONFIDENCE_THRESHOLD", 0.70))

# === RETRY LOGIC ===
MAX_RETRIES_PER_TRADE = int(os.getenv("MAX_RETRIES_PER_TRADE", 3))
MAX_RETRIES           = MAX_RETRIES_PER_TRADE          # ← alias for trade_manager
RETRY_DELAY_SECONDS   = int(os.getenv("RETRY_DELAY_SECONDS", 5))

# === OPTION FILTERS ===
OPTION_TYPE_FILTER        = os.getenv("OPTION_TYPE_FILTER", "both")
MIN_DELTA                 = float(os.getenv("MIN_DELTA", 0.25))
MAX_DELTA                 = float(os.getenv("MAX_DELTA", 0.75))
MAX_THETA                 = float(os.getenv("MAX_THETA", -0.01))
MAX_VEGA                  = float(os.getenv("MAX_VEGA", 0.15))
MAX_GAMMA                 = float(os.getenv("MAX_GAMMA", 0.15))
MIN_VOLUME                = int(os.getenv("MIN_VOLUME", 100))
MIN_OPEN_INTEREST         = int(os.getenv("MIN_OPEN_INTEREST", 100))
MAX_BID_ASK_SPREAD_PCT    = float(os.getenv("MAX_BID_ASK_SPREAD_PCT", 0.15))
MIN_MONEYNESS             = float(os.getenv("MIN_MONEYNESS", 0.90))
MAX_MONEYNESS             = float(os.getenv("MAX_MONEYNESS", 1.10))
ENABLE_DIRECTIONAL_SKEW   = os.getenv("ENABLE_DIRECTIONAL_SKEW", "true").lower() == "true"
DEBUG_OPTION_FILTER       = os.getenv("DEBUG_OPTION_FILTER", "false").lower() == "true"

# === META / PPO TRAINING ===
EPOCHS               = int(os.getenv("EPOCHS", 100))
BATCH_SIZE           = int(os.getenv("BATCH_SIZE", 128))
BUFFER_ALPHA         = float(os.getenv("BUFFER_ALPHA", 0.6))
BUFFER_BETA_START    = float(os.getenv("BUFFER_BETA_START", 0.4))
BUFFER_BETA_INCREMENT = float(os.getenv("BUFFER_BETA_INCREMENT", 0.004))
META_LOG_PATH        = os.getenv("META_LOG_PATH", "meta/meta_log.jsonl")
ENTROPY_COEF_START   = float(os.getenv("ENTROPY_COEF_START", 0.15))
ENTROPY_COEF_END     = float(os.getenv("ENTROPY_COEF_END", 0.03))
META_MODEL_PATH = os.getenv("META_MODEL_PATH", "meta/models/ppo_agent.pth")
META_INFO_PATH = "meta/meta_agent_info.json"
META_STATE_LOOKBACK_MINUTES = 60

# === MACHINE LEARNING ===
CLASSIFIER_MODEL_PATH = "models/xgb_raw.json"
CLASSIFIER_LOG_PATH = "logs/classifier/"
CLASSIFIER_RETRAIN_THRESHOLD = 200  # retrain after 200 new samples

# === DYNAMIC SIZING ===
ENABLE_DYNAMIC_SIZING = os.getenv("ENABLE_DYNAMIC_SIZING", "true").lower() == "true"
MIN_POSITION_SIZE     = float(os.getenv("MIN_POSITION_SIZE", 0.05))
MAX_POSITION_SIZE     = float(os.getenv("MAX_POSITION_SIZE", 0.25))

# === SIMULATION ===
SIMULATION_MODE        = os.getenv("SIMULATION_MODE", "false").lower() == "true"
DEFAULT_SLIPPAGE_BPS   = int(os.getenv("DEFAULT_SLIPPAGE_BPS", 5))
DEFAULT_SPREAD_BPS     = int(os.getenv("DEFAULT_SPREAD_BPS", 8))
SIM_MIN_FILL_DELAY_MS  = int(os.getenv("SIM_MIN_FILL_DELAY_MS", 50))
SIM_MAX_FILL_DELAY_MS  = int(os.getenv("SIM_MAX_FILL_DELAY_MS", 150))

# === EXIT SETTINGS ===
HARD_CLOSE_DAYTRADES_ONLY = os.getenv("HARD_CLOSE_DAYTRADES_ONLY", "false").lower() == "true"

# ───── Economic Event Filters ─────
ECONOMIC_EVENTS = [
    "CPI", "PPI", "Jobs Report", "FOMC", "GDP", "Unemployment",
    "NFP", "Core PCE", "Retail Sales", "Fed Decision"
]

FED_SPEECH_KEYWORDS = [
    "Powell", "FOMC", "Federal Reserve", "interest rate",
    "monetary policy", "inflation", "hawkish", "dovish"
]