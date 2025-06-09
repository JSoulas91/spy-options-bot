import os
import json
import torch
import random
import numpy as np
from datetime import datetime, timedelta

from meta.ppo import PPOAgent
from meta.meta_env import MetaEnv
from meta.meta_agent_info import load_agent_info
from config import META_LOG_PATH
from utils.logger import bot_logger as logger

# Constants
MAX_UPDATES = 1000               # maximum samples to train on
MIN_SAMPLES = 200                # minimum needed to trigger update
CHECKPOINT_PATH = "meta/meta_agent_latest.pth"

def load_recent_experiences(log_path, max_hours=24):
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=max_hours)
    samples = []

    try:
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
    except FileNotFoundError:
        logger.warning(f"Meta log file not found at {log_path}")
        return []
    return samples

def online_update():
    logger.info("[Meta Online Update] Starting PPO online update")

    # Load experience samples
    samples = load_recent_experiences(META_LOG_PATH)
    if len(samples) < MIN_SAMPLES:
        logger.info(f"Only {len(samples)} samples found. Skipping update.")
        return

    # Shuffle and limit batch size
    random.shuffle(samples)
    samples = samples[:MAX_UPDATES]

    # Load agent info (state/action dims)
    info = load_agent_info()
    input_dim = info["state_dim"]
    output_dim = info["action_dim"]

    # Initialize PPO policy
    agent = PPOPolicy(input_dim=input_dim, output_dim=output_dim)
    if os.path.exists(CHECKPOINT_PATH):
        agent.load(CHECKPOINT_PATH)
        logger.info(f"[Meta Online Update] Loaded policy from {CHECKPOINT_PATH}")
    else:
        logger.info("[Meta Online Update] No prior policy found. Starting fresh.")

    # Format training tensors
    states      = torch.tensor([s["state"] for s in samples], dtype=torch.float32)
    actions     = torch.tensor([s["action"] for s in samples], dtype=torch.long)
    rewards     = torch.tensor([s["reward"] for s in samples], dtype=torch.float32)
    next_states = torch.tensor([s["next_state"] for s in samples], dtype=torch.float32)
    dones       = torch.tensor([s["done"] for s in samples], dtype=torch.bool)

    # Run training step
    agent.train_batch(states, actions, rewards, next_states, dones)

    # Save updated model
    agent.save(CHECKPOINT_PATH)
    logger.info(f"[Meta Online Update] Update complete. Saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    online_update()