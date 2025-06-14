import os
import sys
import json
import csv
from typing import List, Dict

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo                   import PPOAgent
from meta.meta_agent_info       import save_meta_agent_dims
from meta.prioritized_buffer    import PrioritizedReplayBuffer
from utils.logger               import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report, send_meta_agent_report
from utils.telegram_utils       import send_telegram_message
from monitor.health_check       import update_status
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

def _extract_td_error(td_err_raw):
    if isinstance(td_err_raw, torch.Tensor):
        return td_err_raw.detach().cpu().tolist()
    elif isinstance(td_err_raw, list):
        return td_err_raw
    elif isinstance(td_err_raw, dict):
        # Flatten any nested dict to first tensor/list found
        for v in td_err_raw.values():
            result = _extract_td_error(v)
            if result is not None:
                return result
    return None

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
        agent.entropy_coef.data *= 0.98
        num_batches = max(1, len(buffer) // BATCH_SIZE)

        for _ in range(num_batches):
            batch, idxs, weights = buffer.sample(BATCH_SIZE, beta)

            states, dirs, confs, rewards, dones = [], [], [], [], []
            for s, a, r, _, done in batch:
                s_vec = _pad_or_trim(s, state_dim)
                states.append(s_vec)
                if isinstance(a, (list, tuple)):
                    dirs.append(int(a[0]))
                    confs.append(float(a[1]))
                else:
                    dirs.append(int(a))
                    confs.append(0.5)
                rewards.append(float(r))
                dones.append(int(bool(done)))

            if not states:
                continue

            rewards_np = np.array(rewards, dtype=np.float32)
            rewards_np = (rewards_np - rewards_np.mean()) / (rewards_np.std() + 1e-8)
            rewards = rewards_np.tolist()

            states_t  = torch.tensor(np.array(states, dtype=np.float32))
            next_t    = states_t.clone()
            dirs_t    = torch.tensor(np.array(dirs, dtype=torch.int64))
            confs_t   = torch.tensor(np.array(confs, dtype=np.float32))
            weights_t = torch.tensor(np.array(weights, dtype=np.float32))
            old_logp  = torch.zeros(len(states_t))

            td_err_raw = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards=rewards,
                dones=dones,
                next_states=next_t,
                old_logp=old_logp,
                weights=weights_t
            )

            td_err_list = _extract_td_error(td_err_raw)
            if td_err_list is None:
                logger.error("Unexpected td_err format: %s", type(td_err_raw))
                continue

            buffer.update_priorities(idxs, td_err_list)
            epoch_rewards.extend(rewards)

            if DEBUG:
                logger.debug("Sampled directions: %s", dirs)
                logger.debug("Sampled confidences: %s", confs)
                logger.debug("Sampled TD errors: %s", td_err_list)

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