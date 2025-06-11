# meta/train_meta_agent.py
import os, sys, json, csv
from typing import List, Dict

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo                   import PPOAgent
from meta.meta_agent_info       import save_meta_agent_dims
from meta.prioritized_buffer    import PrioritizedReplayBuffer
from utils.logger               import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report
from utils.telegram_utils       import send_telegram_message
from monitor.health_check       import update_status
from config import (
    META_LOG_PATH, EPOCHS, BATCH_SIZE,
    BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT
)

CSV_PATH, NOTIFY_EVERY, ACTION_DIM = "meta/reward_history.csv", 10, 3

# ────────────────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _discover_state_dim(rows) -> int:
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list) and len(ms) >= 5:
            return len(ms)
    return -1

def _pad_or_trim(vec, dim):
    if not isinstance(vec, (list, np.ndarray)):
        return [0.0] * dim
    lst = list(vec)
    return lst[:dim] if len(lst) >= dim else lst + [0.0] * (dim - len(lst))

def _prep_buffer(rows, dim):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    for cur in rows:
        ms = cur.get("meta_state")
        if not isinstance(ms, (list, np.ndarray)):
            continue
        st = np.asarray(_pad_or_trim(ms, dim), dtype=np.float32)
        rew = float(cur.get("reward", 0))
        act_raw = cur.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(act_raw, dict):
            act = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5)))
        else:
            try:
                act = (int(act_raw), 0.5)
            except Exception:
                act = (0, 0.5)
        buf.add(st, act, rew, st, True, error=1.0)
    logger.info(f"Buffer ready with {len(buf)} samples.")
    return buf

def _append_csv(ep, val):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([ep, val])

# ────────────────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data."); return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Could not infer state_dim."); return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Buffer too small."); return

    agent = PPOAgent(state_dim=state_dim)
    beta  = BUFFER_BETA_START
    history = []

    for ep in range(1, EPOCHS + 1):
        batch_rewards = []

        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights, *_ = buffer.sample(BATCH_SIZE, beta)

            states_l, next_l, dirs_l, confs_l, rewards_l, dones_l = [], [], [], [], [], []

            for s, a, r, ns, done_flag, *_ in batch:
                vec = _pad_or_trim(s.tolist() if isinstance(s, np.ndarray) else s, state_dim)
                if len(vec) != state_dim:
                    continue
                states_l.append(vec)
                next_l.append(vec)  # bootstrap
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    dirs_l.append(int(a[0]))
                    confs_l.append(float(a[1]))
                else:
                    dirs_l.append(int(float(a)))
                    confs_l.append(0.5)
                rewards_l.append(float(r))
                dones_l.append(int(bool(done_flag)))

            if not states_l:
                continue

            # Ensure alignment
            weights_slice = weights[:len(rewards_l)]

            # tensors
            states_t  = torch.tensor(states_l, dtype=torch.float32)
            next_t    = torch.tensor(next_l,    dtype=torch.float32)
            dirs_t    = torch.tensor(dirs_l,    dtype=torch.long)
            confs_t   = torch.tensor(confs_l,   dtype=torch.float32)
            dones_t   = torch.tensor(dones_l,   dtype=torch.float32)
            weights_t = torch.tensor(weights_slice, dtype=torch.float32)
            old_logp  = torch.zeros(len(rewards_l))

            td_err = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards_l, dones_t,
                next_t, old_logp, weights_t
            )
            buffer.update_priorities(idxs[:len(rewards_l)], td_err)
            batch_rewards.extend(rewards_l)

        avg = float(np.mean(batch_rewards)) if batch_rewards else 0.0
        history.append(avg)
        _append_csv(ep, avg)
        logger.info(f"Epoch {ep}/{EPOCHS}  avg_reward={avg:.4f}")

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep, "avg_reward": avg,
                 "reward_std": float(np.std(batch_rewards)) if batch_rewards else 0},
                history,
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()