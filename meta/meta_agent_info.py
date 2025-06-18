# meta/meta_agent_info.py

import os
import json
from config import META_INFO_PATH
from utils.logger import bot_logger as logger

def save_meta_agent_dims(state_dim: int, action_dim: int):
    info = {"state_dim": state_dim, "action_dim": action_dim}
    try:
        with open(META_INFO_PATH, "w") as f:
            json.dump(info, f, indent=4)
        logger.info(f"📦 Saved meta agent dims → {META_INFO_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to save meta agent dims: {e}")

def get_meta_agent_dims():
    if not os.path.exists(META_INFO_PATH):
        logger.warning(f"⚠️ Meta info not found at {META_INFO_PATH}, using defaults.")
        return 83, 3

    try:
        with open(META_INFO_PATH) as f:
            info = json.load(f)
        sd, ad = int(info.get("state_dim", 83)), int(info.get("action_dim", 3))
        if sd <= 0 or ad <= 0:
            raise ValueError
        return sd, ad
    except Exception as e:
        logger.error(f"❌ Failed to load meta agent dims from {META_INFO_PATH}: {e}")
        return 83, 3  # Safe fallback defaults

# Alias for compatibility
get_meta_agent_info = get_meta_agent_dims