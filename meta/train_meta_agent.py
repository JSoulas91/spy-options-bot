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
MAX_RECENT_SKIP = 300
SEQ_LEN = 20  # sequence length for meta state

def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def _discover_state_dim(rows):
    """
    Look for a list-of-vectors in any of these keys:
      • meta_state        (legacy)
      • meta_entry_state  (current)
      • meta_exit_state   (optional)
    Return the length of 1 vector (the real state_dim), or -1 if not found.
    """
    candidate_keys = ("meta_state", "meta_entry_state", "meta_exit_state")

    for r in rows:
        for k in candidate_keys:
            ms = r.get(k)
            if isinstance(ms, list) and ms and isinstance(ms[0], (list, np.ndarray)):
                return len(ms[0])          # <- FOUND IT ✔
    return -1

def _pad_or_trim(seq, dim, seq_len=SEQ_LEN):
    if not isinstance(seq, list) or not all(isinstance(x, (list, np.ndarray)) for x in seq):
        return np.zeros((seq_len, dim), dtype=np.float32)
    arr = np.array(seq, dtype=np.float32)
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    else:
        pad = np.zeros((seq_len - arr.shape[0], dim), dtype=np.float32)
        return np.vstack([arr, pad])

def _log_classifier_correlation(rows):
    preds, rewards = [], []
    for r in rows:
        cls = r.get("classifier")
        rew = r.get("reward")
        if not cls or rew is None:
            continue
        p = cls.get("trade_success_prob") or cls.get("class_probabilities", [None, None])[1]
        if p is not None:
            preds.append(float(p))
            rewards.append(float(rew))
    if len(preds) >= 10:
        corr = np.corrcoef(preds, rewards)[0, 1]
        logger.info("🔍 Classifier-success vs reward correlation: %.4f", corr)

def _log_entropy_vs_reward(rows):
    entropies, rewards = [], []
    for r in rows:
        cls = r.get("classifier", {})
        ent = cls.get("entropy")
        rew = r.get("reward")
        if ent is not None and rew is not None:
            entropies.append(ent)
            rewards.append(rew)
    if len(entropies) >= 10:
        corr = np.corrcoef(entropies, rewards)[0, 1]
        logger.info("🔍 Classifier-entropy vs reward correlation: %.4f", corr)

def _prep_buffer(rows, dim):
    buf = PrioritizedReplayBuffer(
        capacity=BUFFER_CAPACITY,
        alpha=BUFFER_ALPHA,
        sequence_length=SEQ_LEN,
        state_dim=dim
    )
    rewards = []

    candidate_keys = ("meta_state", "meta_entry_state", "meta_exit_state")

    filtered = rows[-BUFFER_CAPACITY - MAX_RECENT_SKIP:-MAX_RECENT_SKIP] \
               if len(rows) > MAX_RECENT_SKIP else rows
    np.random.shuffle(filtered)

    for row in filtered:
        # ---- NEW: grab whichever meta-state key exists ----
        ms = None
        for k in candidate_keys:
            if k in row:
                ms = row[k]
                break
        if ms is None:
            logger.debug("Skipping row with no meta state key")
            continue
        # ---------------------------------------------------
    
        try:
            st = _pad_or_trim(ms, dim, seq_len=SEQ_LEN)
        except Exception as e:
            logger.warning(f"Skipping malformed meta_state: {e}")
            continue
    
        # ✅ Use shaped_reward if present
        rew = float(row.get("shaped_reward", row.get("reward", 0.0)))
        rewards.append(rew)
    
        act_raw = row.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(act_raw, dict):
            act = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5)))
        else:
            act = (int(act_raw), 0.5)
    
        buf.add(st, act, rew, st, True)

    logger.info("Replay buffer size: %d | Skipped most recent %d rows", len(buf), MAX_RECENT_SKIP)
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

    _log_classifier_correlation(rows)
    _log_entropy_vs_reward(rows)

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Cannot infer state_dim from training data.")
        return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer, all_rewards = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Replay buffer too small to train.")
        return

    rewards_np = np.array(all_rewards)

    # Skip only if truly no signal
    if np.all(rewards_np == 0):
        logger.warning("All rewards are zero. Training skipped.")
        return
    
    # Proceed with normal quantile split for buffer sampling
    high_rew_cutoff = np.percentile(rewards_np, 75)
    low_rew_cutoff = np.percentile(rewards_np, 20)
    reward_spread = high_rew_cutoff - low_rew_cutoff
    
    if reward_spread < 1e-3:  # replace with a small constant if needed
        logger.warning(
            "Training skipped due to low reward diversity: "
            "75th percentile = %.4f, 20th percentile = %.4f, spread = %.4f",
            high_rew_cutoff, low_rew_cutoff, reward_spread
        )
        return
    
    agent = PPOAgent(state_dim=state_dim)
    agent.entropy_coef.data.fill_(ENTROPY_COEF_START)
    decay_rate = (ENTROPY_COEF_END / ENTROPY_COEF_START) ** (1 / EPOCHS)
    logger.info("Entropy decay rate per epoch: %.10f", decay_rate)

    beta = BUFFER_BETA_START
    history_rewards = []

    for epoch in range(1, EPOCHS + 1):
        epoch_rewards = []
        all_td_errors = []

        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch = buffer.sample_balanced(BATCH_SIZE, beta, high_rew_cutoff, low_rew_cutoff)
            states_np = np.array(batch["states"], dtype=np.float32)
            states_t = torch.from_numpy(states_np)
            next_t = states_t.clone()

            dirs, confs = zip(*batch["actions"])
            dirs_t = torch.tensor(dirs, dtype=torch.long)
            confs_t = torch.tensor(confs, dtype=torch.float32)

            rewards_np = np.array(batch["rewards"], dtype=np.float32)
            rewards_np = np.nan_to_num(rewards_np, nan=0.0, posinf=1.0, neginf=-1.0)
            rewards_t = torch.tensor(rewards_np, dtype=torch.float32)

            dones = batch["dones"]
            indices = batch["indices"]
            weights_t = torch.tensor(batch["weights"], dtype=torch.float32)

            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)
            advantages = torch.clamp(advantages, -5, 5)

            conf_mask = (rewards_t > low_rew_cutoff).float()
            confidence_penalty_mask = (rewards_t < low_rew_cutoff).float()

            td_err = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards=rewards_np,
                dones=dones,
                next_states=next_t,
                old_logp=None,
                weights=weights_t,
                advantages=advantages,
                conf_mask=conf_mask,
                confidence_penalty_mask=confidence_penalty_mask
            )

            td_err_list = [float(x) for x in td_err.detach().cpu().flatten()]
            buffer.update_priorities(indices, td_err_list)
            all_td_errors.extend(td_err_list)
            epoch_rewards.extend(rewards_np)

            if DEBUG:
                logger.debug("Sampled dirs: %s", dirs)
                logger.debug("Sampled confs: %s", confs)
                logger.debug("TD errors: %s", td_err_list)

        with torch.no_grad():
            agent.entropy_coef.mul_(decay_rate)

        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        std_reward = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        max_reward = float(np.max(epoch_rewards)) if epoch_rewards else 0.0
        min_reward = float(np.min(epoch_rewards)) if epoch_rewards else 0.0

        history_rewards.append(avg_reward)
        _append_csv(epoch, avg_reward)

        current_lr = agent.optimizer.param_groups[0].get("lr", 0.0)
        if agent.scheduler:
            agent.scheduler.step(avg_reward)

        logger.info(
            "📈 Epoch %d/%d – avg: %.4f  max: %.2f  min: %.2f  std: %.2f  entropy_coef: %.8f  lr: %.8f",
            epoch, EPOCHS, avg_reward, max_reward, min_reward, std_reward,
            agent.entropy_coef.item(), current_lr
        )

        if DEBUG and all_td_errors:
            td_arr = np.array(all_td_errors)
            logger.info("TD error histogram – mean: %.4f | std: %.4f | max: %.4f | min: %.4f",
                        td_arr.mean(), td_arr.std(), td_arr.max(), td_arr.min())

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