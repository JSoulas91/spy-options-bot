# utils/meta_logger.py

import json
import os
from datetime import datetime
from config import META_LOG_PATH

def log_meta_experience(state, action, reward):
    """
    Appends a meta-experience to the log file.
    Each line contains: timestamp, state, action, reward
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "state": state,
        "action": int(action),
        "reward": float(reward)
    }

    try:
        os.makedirs(os.path.dirname(META_LOG_PATH), exist_ok=True)
        with open(META_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"❌ Failed to log meta experience: {e}")