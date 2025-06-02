# meta/meta_agent_info.py

import os
import json
from config import META_INFO_PATH
from utils.logger import bot_logger as logger

def save_meta_agent_dims(state_dim: int, action_dim: int):
    """Save state/action dimensions to meta_agent_info.json."""
    info = {
        "state_dim": state_dim,
        "action_dim": action_dim
    }

    try:
        with open(META_INFO_PATH, 'w') as f:
            json.dump(info, f, indent=4)
        logger.info(f"📦 Saved meta agent dims to {META_INFO_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to save meta agent dims to {META_INFO_PATH}: {e}")

def get_meta_agent_dims():
    """Load state/action dimensions from meta_agent_info.json."""
    if not os.path.exists(META_INFO_PATH):
        raise FileNotFoundError(f"Meta info file not found at {META_INFO_PATH}")

    try:
        with open(META_INFO_PATH, 'r') as f:
            info = json.load(f)
            state_dim = int(info.get('state_dim', -1))
            action_dim = int(info.get('action_dim', -1))

            if state_dim <= 0 or action_dim <= 0:
                raise ValueError(f"Invalid dims in {META_INFO_PATH}: {info}")

            return state_dim, action_dim

    except Exception as e:
        logger.error(f"❌ Failed to load meta agent dims from {META_INFO_PATH}: {e}")
        raise

def meta_agent_dims_exist():
    """Check whether META_INFO_PATH exists and contains valid dims."""
    if not os.path.exists(META_INFO_PATH):
        return False
    try:
        state_dim, action_dim = get_meta_agent_dims()
        return state_dim > 0 and action_dim > 0
    except Exception:
        return False