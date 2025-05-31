# meta/train_meta_agent.py

import os
import json
import numpy as np
from meta.ppo import PPOAgent
from meta.meta_agent_info import save_meta_agent_dims
from config import META_LOG_PATH
from utils.logger import bot_logger as logger
import torch

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

    # Automatically determine state/action dimensions
    state_dim = len(data[0]['state'])
    action_dim = len(data[0]['action'])

    # Save to meta_agent_info.json
    save_meta_agent_dims(state_dim, action_dim)

    # Initialize PPO agent and load existing weights
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA)
    agent.load_model()

    # Convert data to tensors
    states = [torch.tensor(d['state'], dtype=torch.float32) for d in data]
    actions = [int(np.argmax(d['action'])) if isinstance(d['action'], list) else int(d['action']) for d in data]
    rewards = [float(d['reward']) for d in data]
    next_states = [torch.tensor(d['next_state'], dtype=torch.float32) for d in data]
    dones = [bool(d.get('done', False)) for d in data]

    # Estimate values and next_value for return calculation
    with torch.no_grad():
        values = [agent.model.forward(s.unsqueeze(0))[1].item() for s in states]
        next_value = agent.model.forward(next_states[-1].unsqueeze(0))[1].item()

    # Build memory dict for PPOAgent.update()
    memory = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "log_probs": [],
        "values": torch.tensor(values),
        "next_value": torch.tensor(next_value)
    }

    # Precompute log_probs
    for s, a in zip(states, actions):
        probs, _ = agent.model.forward(s.unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(a))
        memory["log_probs"].append(log_prob)

    # Training loop
    for epoch in range(EPOCHS):
        agent.update(memory)
        avg_reward = np.mean(rewards)
        logger.info(f"📈 Epoch {epoch + 1}/{EPOCHS} — Avg Reward: {avg_reward:.4f}")

    # Save updated model
    agent.save_model()
    logger.info("✅ PPO Meta-Agent retraining completed.")

if __name__ == "__main__":
    main()