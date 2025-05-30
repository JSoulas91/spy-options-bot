import os

def prompt_env_var(key, description, default=None, is_secret=False):
    prompt = f"{description} [{default if default is not None else 'REQUIRED'}]: "
    if is_secret:
        import getpass
        value = getpass.getpass(prompt)
    else:
        value = input(prompt)

    if not value and default is not None:
        return default
    elif not value:
        raise ValueError(f"Environment variable '{key}' is required.")
    return value

env_vars = {
    # === ALPACA API ===
    "ALPACA_API_KEY": ("Your Alpaca API Key", None, True),
    "ALPACA_SECRET_KEY": ("Your Alpaca Secret Key", None, True),
    "ALPACA_BASE_URL": ("Alpaca Base URL (paper or live)", "https://paper-api.alpaca.markets", False),

    # === TELEGRAM ===
    "TELEGRAM_BOT_TOKEN": ("Telegram Bot Token", None, True),
    "TELEGRAM_CHAT_ID": ("Telegram Chat ID", None, False),

    # === STRATEGY ===
    "DEFAULT_POSITION_SIZE": ("Default position size (0.1 = 10%)", "0.1", False),
    "MAX_DAY_TRADES": ("Max trades per day", "3", False),
    "AGGRESSIVE_TRADE_SIZE": ("Aggressive trade size (0.15 = 15%)", "0.15", False),
    "MIN_OPTION_EXPIRY_DAYS": ("Minimum expiry in days for options", "7", False),
    "CONFIDENCE_THRESHOLD": ("Standard confidence threshold", "0.75", False),
    "STOP_LOSS_ATR_MULTIPLIER": ("Stop-loss ATR multiplier", "1.5", False),
    "TRAILING_STOP_PERCENT": ("Trailing stop percent (e.g. 0.10)", "0.10", False),
    "PREFERS_LIQUID_OPTIONS": ("Prefer liquid options? (true/false)", "true", False),

    # === TIME CONTROL ===
    "MARKET_OPEN": ("Market open time (ET)", "09:30", False),
    "MARKET_CLOSE": ("Market close time (ET)", "16:00", False),
    "NO_NEW_TRADES_AFTER": ("Cutoff time for new trades (ET)", "15:30", False),

    # === MODES & FLAGS ===
    "USE_AGGRESSIVE_MODE": ("Use aggressive mode? (true/false)", "false", False),

    # === ADVANCED FILTERS ===
    "ENABLE_EVENT_BLACKOUT": ("Block trading during economic events? (true/false)", "true", False),
    "ENABLE_VIX_THROTTLING": ("Enable VIX-based trade throttling? (true/false)", "true", False),
    "ENABLE_FED_SPEAKER_FILTER": ("Block trades during Fed speeches? (true/false)", "true", False),
    "ENABLE_ADAPTIVE_CONFIDENCE": ("Enable adaptive confidence? (true/false)", "true", False),

    # === VIX SETTINGS ===
    "VIX_MAX_THRESHOLD": ("Max VIX before disabling trades", "30.0", False),
    "VIX_MODERATE_THRESHOLD": ("Moderate VIX level (confidence raised)", "25.0", False),
    "VIX_SAFE_FOR_SWING": ("Max VIX for weekend swing eligibility", "20.0", False),

    # === ADAPTIVE CONFIDENCE ===
    "BASE_CONFIDENCE_THRESHOLD": ("Base confidence threshold (normal)", "0.55", False),
    "OPENING_RANGE_THRESHOLD": ("Confidence threshold for early trades", "0.50", False),
    "HIGH_VIX_THRESHOLD": ("Confidence threshold when VIX is high", "0.65", False),
    "SWING_TRADE_CONFIDENCE_THRESHOLD": ("Min confidence for swing trades", "0.70", False),

    # === RETRY SETTINGS ===
    "MAX_RETRIES_PER_TRADE": ("Max retry attempts per trade", "3", False),
    "RETRY_DELAY_SECONDS": ("Delay between retries (in seconds)", "5", False),
}

print("📄 Generating .env file for your SPY options bot...\n")
env_lines = []

for key, (desc, default, secret) in env_vars.items():
    try:
        val = prompt_env_var(key, desc, default, secret)
        env_lines.append(f"{key}={val}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)

with open(".env", "w") as f:
    f.write("\n".join(env_lines))

print("\n✅ .env file created successfully.")