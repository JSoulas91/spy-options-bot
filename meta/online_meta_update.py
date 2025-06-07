# meta/online_meta_update.py

import os
import json
import torch
import random
import numpy as np
from datetime import datetime, timedelta

from meta.meta_agent import PPOAgent
from meta.meta_env import MetaEnv
from meta.meta_agent_info import load_agent_info
from config import META_LOG_PATH
from utils.logger import bot_logger as logger

# Constants
MAX_UPDATES = 1000               # maximum steps per retrain
MIN_SAMPLES = 200                # min new experiences before retrain
CHECKPOINT_PATH = "meta/meta_agent_latest.pth"

def load_recent_experiences(log_path, max_hours=24):
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=max_hours)
    samples = []

    with open(log_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ts = obj.get("timestamp")
                if not ts:
                    samples.append(obj)
                    continue
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
                if t >= cutoff:
                    samples.append(obj)
            except Exception:
                continue
    return samples

def online_update():
    logger.info("[Meta Online Update] Starting online PPO update")

    # Load recent experiences
    if not os.path.exists(META_LOG_PATH):
        logger.warning("No meta log file found.")
        return

    samples = load_recent_experiences(META_LOG_PATH)
    if len(samples) < MIN_SAMPLES:
        logger.info(f"Not enough samples ({len(samples)} found). Skipping update.")
        return

    # Shuffle and trim to MAX_UPDATES
    random.shuffle(samples)
    samples = samples[:MAX_UPDATES]

    # Load PPO agent and info
    info = load_agent_info()
    agent = PPOAgent(state_dim=info["state_dim"], action_dim=info["action_dim"])

    if os.path.exists(CHECKPOINT_PATH):
        agent.load(CHECKPOINT_PATH)
        logger.info(f"Loaded existing agent from {CHECKPOINT_PATH}")

    # Format training batch
    states      = torch.tensor([s["state"]], dtype=torch.float32).squeeze()
    actions     = torch.tensor([s["action"] for s in samples], dtype=torch.long)
    rewards     = torch.tensor([s["reward"] for s in samples], dtype=torch.float32)
    next_states = torch.tensor([s["next_state"]], dtype=torch.float32).squeeze()
    dones       = torch.tensor([s["done"] for s in samples], dtype=torch.bool)

    # Run training step
    agent.train_batch(states, actions, rewards, next_states, dones)

    # Save updated weights
    agent.save(CHECKPOINT_PATH)
    logger.info(f"Online meta-agent update complete. Saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    online_update()