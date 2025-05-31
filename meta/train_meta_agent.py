# meta/train_meta_agent.py

import os
import json
import numpy as np
from meta.ppo import PPO
from config import META_LOG_PATH
from utils.logger import bot_logger as logger

# === Hyperparameters ===
GAMMA = 0.99
LR = 0.0003
EPOCHS = 20
BATCH_SIZE = 32

def load_logged_data():
    data = []
    if not os.path.exists(META_LOG_PATH):
        logger.warning(f"⚠️ No meta log found at {META_LOG_PATH}")
        return data

    with open(META_LOG_PATH, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                logger.warning("⚠️ Skipped invalid JSON line in log.")
    return data

def main():
    logger.info("🎯 Starting PPO Meta-Agent Retraining from Logs...")

    data = load_logged_data()
    if not data:
        logger.warning("❌ No data available for training. Aborting.")
        return

    # Extract dimensions
    state_dim = len(data[0]['state'])
    action_dim = len(data[0]['action'])

    # Initialize PPO agent
    agent = PPO(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA)

    # Convert data to training batches
    states = np.array([d['state'] for d in data])
    actions = np.array([d['action'] for d in data])
    rewards = np.array([d['reward'] for d in data])
    next_states = np.array([d['next_state'] for d in data])
    dones = np.array([d.get('done', False) for d in data])

    # Add to PPO buffer
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        agent.buffer.append((s, a, r, ns, d))

    # Train for a few epochs
    for epoch in range(EPOCHS):
        agent.train(batch_size=BATCH_SIZE)
        avg_reward = np.mean(rewards)
        logger.info(f"📈 Epoch {epoch+1}/{EPOCHS} — Avg Reward: {avg_reward:.4f}")

    logger.info("✅ PPO Meta-Agent retraining completed.")

if __name__ == "__main__":
    main()