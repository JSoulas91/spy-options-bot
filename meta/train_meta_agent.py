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
    BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT,
    ENTROPY_COEF_START, ENTROPY_COEF_END
)

CSV_PATH = "meta/reward_history.csv"
NOTIFY_EVERY = 10
BUFFER_CAPACITY = 30000
ACTION_DIM = 3
DEBUG = True
MIN_HIGH_REWARD = 1.5  # Only train if at least one good reward exists

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
    rewards = []

    # 🔀 Shuffle before adding to buffer to avoid recency bias
    recent_rows = rows[-BUFFER_CAPACITY:]
    np.random.shuffle(recent_rows)

    for row in recent_rows:
        ms = row.get("meta_state")
        if not isinstance(ms, (list, np.ndarray)):
            continue
        st = np.asarray(_pad_or_trim(ms, dim), dtype=np.float32)

        rew = float(row.get("reward", 0))
        rewards.append(rew)

        act_raw = row.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(act_raw, dict):
            act = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5)))
        else:
            act = (int(act_raw), 0.5)

        buf.add(st, act, rew, st, True)

    logger.info("Replay buffer populated: %d samples", len(buf))
    logger.info("Reward stats — avg: %.4f, std: %.4f, max: %.2f, min: %.2f",
                np.mean(rewards), np.std(rewards), np.max(rewards), np.min(rewards))
    return buf, rewards

def _append_csv(epoch_idx, avg_r):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch", "avg_reward"])
        writer.writerow([epoch_idx, avg_r])

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
    buffer, all_rewards = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Replay buffer too small to train.")
        return

    # 🔍 Skip training if no meaningful high-quality samples
    if np.max(all_rewards) < MIN_HIGH_REWARD:
        logger.warning("No high-reward trades (reward > %.2f). Training skipped.", MIN_HIGH_REWARD)
        return

    # Reward stratification for balanced sampling
    rewards_np = np.array(all_rewards)
    high_rew_cutoff = np.percentile(rewards_np, 70)
    low_rew_cutoff = np.percentile(rewards_np, 30)
    if high_rew_cutoff <= low_rew_cutoff:
        logger.warning("Insufficient reward spread. Training skipped.")
        return

    agent = PPOAgent(state_dim=state_dim)
    agent.entropy_coef.data.fill_(ENTROPY_COEF_START)
    decay_rate = (ENTROPY_COEF_END / ENTROPY_COEF_START) ** (1 / EPOCHS)
    logger.info("Entropy decay rate per epoch: %.10f", decay_rate)

    beta = BUFFER_BETA_START
    history_rewards = []

    for epoch in range(1, EPOCHS + 1):
        epoch_rewards = []
        num_batches = max(1, len(buffer) // BATCH_SIZE)

        for _ in range(num_batches):
            batch = buffer.sample_balanced(BATCH_SIZE, beta, high_rew_cutoff, low_rew_cutoff)
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            dones = batch['dones']
            indices = batch['indices']
            weights = batch['weights']

            dirs, confs = zip(*[(int(a[0]), float(a[1])) for a in actions])
            states_t = torch.tensor(states, dtype=torch.float32)
            next_t   = states_t.clone()
            dirs_t   = torch.tensor(dirs, dtype=torch.long)
            confs_t  = torch.tensor(confs, dtype=torch.float32)
            weights_t = torch.tensor(weights, dtype=torch.float32)

            td_err = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards=rewards,
                dones=dones,
                next_states=next_t,
                old_logp=None,
                weights=weights_t
            )

            td_err_list = td_err.detach().cpu().tolist() if hasattr(td_err, "detach") else list(td_err)
            buffer.update_priorities(indices, td_err_list)
            epoch_rewards.extend(rewards)

            if DEBUG:
                logger.debug("Sampled dirs: %s", dirs)
                logger.debug("Sampled confs: %s", confs)
                logger.debug("TD errors: %s", td_err_list)
                logger.debug("Entropy coef (pre-decay): %.8f", agent.entropy_coef.item())

        with torch.no_grad():
            agent.entropy_coef.mul_(decay_rate)

        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        std_reward = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        max_reward = float(np.max(epoch_rewards)) if epoch_rewards else 0.0
        min_reward = float(np.min(epoch_rewards)) if epoch_rewards else 0.0

        history_rewards.append(avg_reward)
        _append_csv(epoch, avg_reward)

        current_lr = agent.optimizer.param_groups[0].get("lr", 0.0)

        if hasattr(agent, "scheduler") and agent.scheduler:
            agent.scheduler.step(avg_reward)

        logger.info(
            "📈 Epoch %d/%d – avg: %.4f  max: %.2f  min: %.2f  std: %.2f  entropy_coef: %.8f  lr: %.8f",
            epoch, EPOCHS, avg_reward, max_reward, min_reward, std_reward,
            agent.entropy_coef.item(), current_lr
        )

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)

        if epoch % NOTIFY_EVERY == 0:
            send_training_report(
                {
                    "epoch": epoch,
                    "avg_reward": avg_reward,
                    "reward_std": std_reward,
                    "reward_max": max_reward,
                    "reward_min": min_reward,
                    "entropy_coef": agent.entropy_coef.item(),
                    "learning_rate": current_lr
                },
                history_rewards
            )

    agent.save()
    update_status("last_ppo")

    try:
        send_meta_agent_report()
    except Exception as e:
        logger.error("Meta agent report failed: %s", str(e))

    send_telegram_message("⭐ Dual-head PPO training completed.")

if __name__ == "__main__":
    train()