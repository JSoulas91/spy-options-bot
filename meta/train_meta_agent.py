import os
import json
import torch
import numpy as np
from datetime import datetime
from config import META_LOG_PATH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from meta.ppo import PPOAgent
from meta.meta_agent_info import save_meta_agent_dims
import requests

# === Hyperparameters ===
GAMMA = 0.99
LR = 3e-4
EPOCHS = 20
BATCH_SIZE = 32
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
    logger.info("🎯 Starting PPO Meta-Agent Retraining from Logs...")
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

        states = [torch.tensor(d['state'], dtype=torch.float32) for d in data]
        actions = [int(np.argmax(d['action'])) if isinstance(d['action'], list) else int(d['action']) for d in data]
        rewards = [float(d['reward']) for d in data]
        next_states = [torch.tensor(d['next_state'], dtype=torch.float32) for d in data]
        dones = [bool(d.get('done', False)) for d in data]

        with torch.no_grad():
            values = [agent.model(s.unsqueeze(0))[1].item() for s in states]
            next_value = agent.model(next_states[-1].unsqueeze(0))[1].item()

        memory = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": [],
            "values": torch.tensor(values),
            "next_value": torch.tensor(next_value)
        }

        for s, a in zip(states, actions):
            probs, _ = agent.model(s.unsqueeze(0))
            dist = torch.distributions.Categorical(probs)
            memory["log_probs"].append(dist.log_prob(torch.tensor(a)))

        for epoch in range(EPOCHS):
            agent.update(memory)
            avg_reward = np.mean(rewards)
            logger.info(f"📈 Epoch {epoch + 1}/{EPOCHS} — Avg Reward: {avg_reward:.4f}")

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