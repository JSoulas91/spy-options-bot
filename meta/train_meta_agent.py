# train_meta_agent.py

import os
import json
import numpy as np
import torch
import random
import csv
from meta.ppo import PPOAgent
from meta.meta_state import build_meta_state_from_log
from meta.reward_shaper import compute_shaped_reward
from meta.meta_agent_info import save_meta_agent_dims
from meta.prioritized_buffer import PrioritizedReplayBuffer
from config import META_LOG_PATH, EPOCHS, BATCH_SIZE, BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT
from utils.logger import bot_logger as logger
from utils.telegram import send_telegram_message

REWARD_TRACKING_PATH = "meta/reward_history.csv"

def load_meta_data():
    if not os.path.exists(META_LOG_PATH):
        logger.warning(f"⚠️ No meta training data found at {META_LOG_PATH}")
        return []

    with open(META_LOG_PATH, "r") as f:
        lines = f.readlines()

    data = [json.loads(line.strip()) for line in lines if line.strip()]
    return data

def preprocess_data(data):
    buffer = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)

    for i in range(len(data) - 1):
        current = data[i]
        next_item = data[i + 1]

        state = build_meta_state_from_log(current)
        next_state = build_meta_state_from_log(next_item)
        action = current.get("meta_action", 0)
        reward = compute_shaped_reward(current)
        done = current.get("done", False)

        buffer.add(state, action, reward, next_state, done)

    return buffer

def append_reward_to_csv(epoch, reward):
    header_needed = not os.path.exists(REWARD_TRACKING_PATH)
    with open(REWARD_TRACKING_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["epoch", "avg_reward"])
        writer.writerow([epoch, reward])

def train():
    logger.info("🚀 Starting PPO meta-agent training...")
    data = load_meta_data()
    if not data:
        logger.warning("❌ No data to train on.")
        return

    save_meta_agent_dims(data[0])  # Persist dims for loading
    buffer = preprocess_data(data)

    agent = PPOAgent()
    beta = BUFFER_BETA_START
    prev_avg_reward = float("-inf")

    for epoch in range(EPOCHS):
        epoch_rewards = []

        for _ in range(len(buffer) // BATCH_SIZE):
            batch, indices, weights = buffer.sample(BATCH_SIZE, beta)

            states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            actions = [b[1] for b in batch]
            rewards = [b[2] for b in batch]
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
            dones = [b[4] for b in batch]
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

            td_errors = agent.train_step(states, actions, rewards, dones, next_states, weights_tensor)

            epoch_rewards.extend(rewards)
            buffer.update_priorities(indices, td_errors)

        avg_reward = np.mean(epoch_rewards)
        logger.info(f"📈 Epoch {epoch + 1}/{EPOCHS} — Avg Reward: {avg_reward:.4f}")

        append_reward_to_csv(epoch + 1, avg_reward)
        agent.adjust_entropy()

        if epoch > 0 and avg_reward < prev_avg_reward:
            agent.adjust_learning_rate(agent.optimizer, factor=0.9)

        prev_avg_reward = avg_reward
        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)

    agent.save()
    send_telegram_message(f"✅ Meta-agent training complete. Avg reward: {avg_reward:.4f}")
    logger.info("🎉 PPO meta-agent training finished.")

if __name__ == "__main__":
    train()