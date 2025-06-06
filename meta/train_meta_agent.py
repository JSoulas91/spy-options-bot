# train_meta_agent.py  ── FULL UPDATED FILE
import os, json, csv, time
from typing import List, Dict

import numpy as np
import torch

from meta.ppo                 import PPOAgent
from meta.meta_state          import build_meta_state_from_log
from meta.reward_shaper       import compute_shaped_reward
from meta.meta_agent_info     import save_meta_agent_dims
from meta.prioritized_buffer  import PrioritizedReplayBuffer

from utils.logger             import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report
from utils.telegram           import send_telegram_message
from monitor.health_check     import update_status
from config import (
    META_LOG_PATH,
    EPOCHS,                 # e.g. 200
    BATCH_SIZE,             # e.g. 64
    BUFFER_ALPHA,
    BUFFER_BETA_START,
    BUFFER_BETA_INCREMENT,
)

REWARD_TRACKING_PATH = "meta/reward_history.csv"
NOTIFY_EVERY = 10          # epochs per telegram update

# ────────────────────────────────────────────────────────────
def load_meta_data() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        logger.warning(f"⚠️ No meta data at {META_LOG_PATH}")
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]

def preprocess_data(rows):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    for i in range(len(rows) - 1):
        cur, nxt = rows[i], rows[i + 1]

        state      = build_meta_state_from_log(cur)
        next_state = build_meta_state_from_log(nxt)

        raw_action = cur.get("meta_action", 0)
        action     = int(np.argmax(raw_action)) if isinstance(raw_action, list) else int(raw_action)
        reward     = compute_shaped_reward(cur)
        done       = cur.get("done", False)

        buf.add(state, action, reward, next_state, done)
    logger.info(f"[Meta] Loaded {len(buf)} experiences.")
    return buf

def append_reward(epoch: int, reward: float):
    first = not os.path.exists(REWARD_TRACKING_PATH)
    with open(REWARD_TRACKING_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([epoch, reward])

def normalize_rewards(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return list((arr - arr.mean()) / (arr.std() + 1e-8))

def sharpe_ratio(series):
    if len(series) < 2: return 0.0
    r = np.asarray(series, float)
    return (r.mean() / (r.std() + 1e-9)) * np.sqrt(len(r))

# ────────────────────────────────────────────────────────────
def train():
    logger.info("🚀 Starting PPO meta‑agent training …")
    update_status("last_ppo_attempt")

    rows = load_meta_data()
    if not rows:
        logger.warning("❌ No training samples.")
        return

    save_meta_agent_dims(rows[0])
    buffer = preprocess_data(rows)

    agent  = PPOAgent(); agent.train_mode()
    beta   = BUFFER_BETA_START
    reward_history = []

    for epoch in range(1, EPOCHS + 1):
        batch_rewards, batch_actions = [], []

        for _ in range(len(buffer) // BATCH_SIZE):
            batch, idxs, weights = buffer.sample(BATCH_SIZE, beta)

            states       = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            actions      = [b[1] for b in batch]
            raw_rewards  = [b[2] for b in batch]
            rewards_norm = normalize_rewards(raw_rewards)
            next_states  = torch.tensor([b[3] for b in batch], dtype=torch.float32)
            dones        = [b[4] for b in batch]
            w_tensor     = torch.tensor(weights, dtype=torch.float32)

            td_err = agent.train_step(states, actions, rewards_norm, dones, next_states, w_tensor)
            buffer.update_priorities(idxs, td_err)

            batch_rewards.extend(raw_rewards)
            batch_actions.extend(actions)

        avg_r   = float(np.mean(batch_rewards))
        std_r   = float(np.std(batch_rewards))
        reward_history.append(avg_r)
        append_reward(epoch, avg_r)

        sharpe  = sharpe_ratio(batch_rewards)
        reject_rate = batch_actions.count(0) / max(1, len(batch_actions))
        agent.adjust_entropy()
        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)

        logger.info(f"📈 Epoch {epoch}/{EPOCHS}  AvgR={avg_r:.4f}  Sharpe≈{sharpe:.2f}")

        # Telegram every N epochs
        if epoch % NOTIFY_EVERY == 0:
            stats = {
                "epoch":        epoch,
                "avg_reward":   avg_r,
                "reward_std":   std_r,
                "sharpe":       sharpe,
                "reject_rate":  reject_rate,
                "avg_duration": 0.0,        # placeholder – fill if you track it
                "avg_pnl":      0.0,        #   "
            }
            send_training_report(stats, reward_history)

    agent.save()
    update_status("last_ppo")
    send_telegram_message(f"✅ Meta‑agent training done. Final avg reward {reward_history[-1]:.4f}")
    logger.info("🎉 PPO training complete.")

if __name__ == "__main__":
    train()