# meta/train_meta_agent.py

import os
import json
import numpy as np

from meta.meta_env import MetaEnv
from meta.ppo import PPO
from meta.meta_agent import load_meta_state, save_meta_state
from utils.logger import bot_logger as logger

# === Hyperparameters ===
EPISODES = 500
TIMESTEPS = 100
GAMMA = 0.99
LR = 0.0003
UPDATE_FREQ = 20

def main():
    logger.info("🎯 Starting PPO Meta-Agent Training...")

    # Load state
    state_dict = load_meta_state()
    env = MetaEnv(state_dict)

    # Get state & action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create PPO agent
    agent = PPO(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA)

    # Track training rewards
    all_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(TIMESTEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            if done:
                break

        all_rewards.append(total_reward)

        # PPO Policy update
        if (episode + 1) % UPDATE_FREQ == 0:
            agent.train()
            logger.info(f"🧠 Episode {episode + 1}/{EPISODES} — Avg Reward: {np.mean(all_rewards[-UPDATE_FREQ:]):.2f}")

    # Final model output (optional: save to disk)
    logger.info("✅ PPO Meta-Agent training completed.")
    logger.info(f"📊 Final Avg Reward: {np.mean(all_rewards[-50:]):.2f}")

    # Save updated meta state
    save_meta_state(env.state)

if __name__ == "__main__":
    main()