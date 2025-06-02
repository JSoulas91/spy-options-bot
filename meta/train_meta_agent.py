import os
import json
import torch
import numpy as np
from datetime import datetime
from config import META_LOG_PATH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from meta.ppo import PPOAgent
from meta.meta_agent_info import save_meta_agent_dims
from meta.prioritized_buffer import PrioritizedReplayBuffer
import requests

# === Hyperparameters ===
GAMMA = 0.99
LR = 3e-4
EPOCHS = 20
BATCH_SIZE = 32
BUFFER_ALPHA = 0.6
BUFFER_BETA_START = 0.4
BUFFER_BETA_INCREMENT = 0.01
CHECKPOINT_DIR = "meta/checkpoints"
KEEP_LAST_N_CHECKPOINTS = 7
KEEP_LAST_N_DAYS_LOGS = 7

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def notify_telegram(message):
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            requests.post(url, json=payload)
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")

def cleanup_checkpoints():
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ppo_checkpoint_epoch_")],
        key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x))
    )
    for old_ckpt in checkpoints[:-KEEP_LAST_N_CHECKPOINTS]:
        os.remove(os.path.join(CHECKPOINT_DIR, old_ckpt))

def cleanup_old_logs():
    now = datetime.now()
    logs_path = "./logs"
    if not os.path.exists(logs_path): return
    for f in os.listdir(logs_path):
        fp = os.path.join(logs_path, f)
        if os.path.isfile(fp) and (now - datetime.fromtimestamp(os.path.getmtime(fp))).days > KEEP_LAST_N_DAYS_LOGS:
            os.remove(fp)

def load_logged_data():
    data = []
    if not os.path.exists(META_LOG_PATH):
        logger.warning(f"⚠️ No meta log found at {META_LOG_PATH}")
        return data
    with open(META_LOG_PATH, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning("⚠️ Skipped invalid JSON line in log.")
    return data

def main():
    logger.info("🎯 Starting PPO Meta-Agent Retraining with PER...")

    try:
        data = load_logged_data()
        if not data:
            msg = "❌ No data available for training. Aborting."
            logger.warning(msg)
            notify_telegram(msg)
            return

        state_dim = len(data[0]['state'])
        action_dim = len(data[0]['action']) if isinstance(data[0]['action'], list) else 2
        save_meta_agent_dims(state_dim, action_dim)

        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=LR, gamma=GAMMA)
        buffer = PrioritizedReplayBuffer(capacity=len(data), alpha=BUFFER_ALPHA)

        for d in data:
            state = torch.tensor(d['state'], dtype=torch.float32)
            next_state = torch.tensor(d['next_state'], dtype=torch.float32)
            reward = float(d['reward'])
            done = bool(d.get('done', False))
            action = int(np.argmax(d['action'])) if isinstance(d['action'], list) else int(d['action'])

            with torch.no_grad():
                _, value = agent.model(state.unsqueeze(0))
                _, next_value = agent.model(next_state.unsqueeze(0))
                td_error = reward + GAMMA * next_value.item() * (1 - int(done)) - value.item()

            buffer.add({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            }, priority=abs(td_error))

        beta = BUFFER_BETA_START

        for epoch in range(EPOCHS):
            epoch_rewards = []

            for _ in range(len(buffer) // BATCH_SIZE):
                samples, indices, weights = buffer.sample(BATCH_SIZE, beta=beta)
                if not samples:
                    continue

                states = torch.stack([s['state'] for s in samples])
                actions = torch.tensor([s['action'] for s in samples])
                rewards = torch.tensor([s['reward'] for s in samples], dtype=torch.float32)
                dones = torch.tensor([s['done'] for s in samples], dtype=torch.float32)
                next_states = torch.stack([s['next_state'] for s in samples])
                weights_tensor = torch.tensor(weights, dtype=torch.float32)

                td_errors = agent.train_step(states, actions, rewards, dones, next_states, weights_tensor)

                for i, td in zip(indices, td_errors):
                    buffer.update(i, abs(td.item()))

                epoch_rewards.extend(rewards.numpy())

            avg_reward = np.mean(epoch_rewards)
            logger.info(f"📈 Epoch {epoch + 1}/{EPOCHS} — Avg Reward: {avg_reward:.4f}")
            beta = min(1.0, beta + BUFFER_BETA_INCREMENT)

        agent.save()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_checkpoint_epoch_{timestamp}.pt")
        torch.save(agent.model.state_dict(), checkpoint_path)
        logger.info(f"✅ Saved checkpoint to {checkpoint_path}")

        cleanup_checkpoints()
        cleanup_old_logs()

        notify_telegram(f"✅ Meta-Agent training complete — Avg Reward: {avg_reward:.4f}")

    except Exception as e:
        logger.error(f"❌ PPO training failed: {e}")
        notify_telegram(f"❌ Meta-Agent training failed: {e}")

if __name__ == "__main__":
    main()