import os
import json
import time
from datetime import datetime, timedelta
from utils.logger import bot_logger
from utils.telegram_utils import send_telegram_message

STATUS_FILE = "monitor/last_status.json"
MAX_MINUTES_SINCE_LAST_TRADE = 60 * 6  # 6 hours max gap
MAX_MINUTES_SINCE_LAST_RETRAIN = 60 * 24  # 1 day
MAX_MINUTES_SINCE_LAST_PPO_TRAINING = 60 * 24  # 1 day

def _load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

def update_status(event_type: str, status_value: str = "ok"):
    status = _load_status()
    status[event_type] = {
        "status": status_value,
        "timestamp": datetime.utcnow().isoformat()
    }
    _save_status(status)
    bot_logger.info(f"✅ Updated status: {event_type} → {status_value}")

def check_health():
    status = _load_status()
    now = datetime.utcnow()

    alerts = []

    def check_time(event, max_minutes, label):
        last_time_str = status.get(event)
        if not last_time_str:
            alerts.append(f"❗ No recorded {label} status.")
            return

        last_time = datetime.fromisoformat(last_time_str)
        delta = now - last_time
        if delta > timedelta(minutes=max_minutes):
            alerts.append(f"⚠️ {label} outdated by {delta}.")

    check_time("last_trade", MAX_MINUTES_SINCE_LAST_TRADE, "Trade")
    check_time("last_retrain", MAX_MINUTES_SINCE_LAST_RETRAIN, "Retrain")
    check_time("last_ppo", MAX_MINUTES_SINCE_LAST_PPO_TRAINING, "PPO Training")

    if alerts:
        message = "🚨 <b>Bot Health Alert</b>\n\n" + "\n".join(alerts)
        send_telegram_message(message)
        bot_logger.warning("Health check triggered alerts.")
    else:
        bot_logger.info("✅ Bot health check passed.")