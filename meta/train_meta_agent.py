import os
import sys
import json
import csv
from typing import List, Dict

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo import PPOAgent
from meta.meta_agent_info import save_meta_agent_dims
from meta.prioritized_buffer import PrioritizedReplayBuffer
from utils.logger import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report, send_meta_agent_report
from utils.telegram_utils import send_telegram_message
from monitor.health_check import update_status
from config import (
    META_LOG_PATH, EPOCHS, BATCH_SIZE,
    BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT
)

# ─── Config ───────────────────────────────────────────────────────────────
CSV_PATH = "meta/reward_history.csv"
NOTIFY_EVERY = 10
BUFFER_CAPACITY = 10_000
ACTION_DIM = 3
REWARD_EXPONENT = 1.5
DEBUG = True

# ─── Helpers ─────────────────────────────────────────────────────────────

def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def _discover_state_dim(rows):
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list) and len(ms) >= 5:
            return len(ms)
    return -1

def _pad_or_trim(vec, dim):
    if not isinstance(vec, (list, np.ndarray)):
        return [0.0] * dim
    lst = list(vec)
    if len(lst) >= dim:
        return lst[:dim]
    return lst + [0.0] * (dim - len(lst))

def _prep_buffer(rows, dim):
    buf = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY, alpha=BUFFER_ALPHA)
    for row in rows:
        ms = row.get("meta_state")
        if not isinstance(ms, (list, np.ndarray)):
            continue
        st = np.asarray(_pad_or_trim(ms, dim), dtype=np.float32)
        rew = float(row.get("reward", 0))
        rew = np.sign(rew) * (abs(rew) ** REWARD_EXPONENT)
        act_raw = row.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(act_raw, dict):
            act = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5)))
        else:
            act = (int(act_raw), 0.5)
        buf.add(st, act, rew, st, True)
    logger.info("Replay buffer populated: %d samples", len(buf))
    return buf

def _append_csv(epoch_idx, avg_r):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch", "avg_reward"])
        writer.writerow([epoch_idx, avg_r])

# ─── Training ────────────────────────────────────────────────────────────

def train():
    logger.info("🚀 PPO meta-agent training started")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data found.")
        return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Cannot infer state_dim from training data.")
        return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Replay buffer too small to train.")
        return

    agent = PPOAgent(state_dim=state_dim)
    beta = BUFFER_BETA_START
    history_rewards = []

    for epoch in range(1, EPOCHS + 1):
        epoch_rewards = []

        # Optional: gentle entropy decay
        agent.entropy_coef.data *= 0.995

        num_batches = max(1, len(buffer) // BATCH_SIZE)
        for _ in range(num_batches):
            batch = buffer.sample(BATCH_SIZE, beta)
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            dones = batch['dones']
            indices = batch['indices']
            weights = batch['weights']

            dirs, confs = [], []
            for a in actions:
                dirs.append(int(a[0]))
                confs.append(float(a[1]))

            states_t  = torch.tensor(states, dtype=torch.float32)
            next_t    = states_t.clone()
            dirs_t    = torch.tensor(dirs, dtype=torch.long)
            confs_t   = torch.tensor(confs, dtype=torch.float32)
            weights_t = torch.tensor(weights, dtype=torch.float32)
            old_logp  = None  # No KL term during offline training

            td_err = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards=rewards,
                dones=dones,
                next_states=next_t,
                old_logp=old_logp,
                weights=weights_t
            )

            if isinstance(td_err, dict):
                td_err = td_err.get("td_error", [])

            if hasattr(td_err, "detach"):
                td_err_list = td_err.detach().cpu().tolist()
            else:
                td_err_list = list(td_err) if isinstance(td_err, (list, np.ndarray)) else [0.0] * BATCH_SIZE

            buffer.update_priorities(indices, td_err_list)
            epoch_rewards.extend(rewards)

            if DEBUG:
                logger.debug("Sampled dirs: %s", dirs)
                logger.debug("Sampled confs: %s", confs)
                logger.debug("TD errors: %s", td_err_list)

        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        std_reward = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        max_reward = float(np.max(epoch_rewards)) if epoch_rewards else 0.0
        min_reward = float(np.min(epoch_rewards)) if epoch_rewards else 0.0

        history_rewards.append(avg_reward)
        _append_csv(epoch, avg_reward)

        logger.info(
            "📈 Epoch %d/%d – avg: %.4f  max: %.2f  min: %.2f  std: %.2f",
            epoch, EPOCHS, avg_reward, max_reward, min_reward, std_reward
        )

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)

        if epoch % NOTIFY_EVERY == 0:
            send_training_report(
                {
                    "epoch": epoch,
                    "avg_reward": avg_reward,
                    "reward_std": std_reward,
                    "reward_max": max_reward,
                    "reward_min": min_reward
                },
                history_rewards
            )

    agent.save()
    update_status("last_ppo")

    try:
        send_meta_agent_report()
    except Exception as e:
        logger.error("Meta agent report failed: %s", str(e))

    send_telegram_message("✅ Dual-head PPO training completed.")

if __name__ == "__main__":
    train()