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
    """
    Return (state_dim, action_dim) for the meta agent.
    If saved info is missing or invalid, fallback to dynamic dummy detection.
    """
    if os.path.exists(META_INFO_PATH):
        try:
            with open(META_INFO_PATH) as f:
                info = json.load(f)
            sd, ad = int(info.get("state_dim", -1)), int(info.get("action_dim", -1))
            if sd > 0 and ad > 0:
                return sd, ad
            logger.warning("⚠️ Invalid dims in meta info file, using fallback detection...")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load meta info: {e} — falling back to dynamic detection.")

    # ✅ DYNAMIC fallback: use a dummy meta state to compute true shape
    try:
        from meta.meta_state import build_meta_state_for_entry

        dummy_state = build_meta_state_for_entry(
            price=500,
            vix=15.0,
            volume=1e6,
            timestamp="2023-01-01 09:30:00",
            features={},
            classifier_outputs={}
        )
        state_dim = dummy_state.shape[0]
        action_dim = 3  # [long, short, hold]
        logger.info(f"🧠 Fallback detected meta agent dims: state_dim={state_dim}, action_dim={action_dim}")
        return state_dim, action_dim
    except Exception as e:
        logger.error(f"❌ Could not dynamically determine meta agent dims: {e}")
        raise RuntimeError("Failed to get meta agent dims.")


# Alias for compatibility
get_meta_agent_info = get_meta_agent_dims