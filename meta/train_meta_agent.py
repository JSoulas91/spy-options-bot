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

CSV_PATH, NOTIFY_EVERY   = "meta/reward_history.csv", 10
BUFFER_CAPACITY          = 10_000        # replay buffer size
ACTION_DIM               = 3             # categorical actions 0/1/2

# ╭──────────────────────────────────────────────────────────╮
# │ Helpers                                                 │
# ╰──────────────────────────────────────────────────────────╯
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

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
    return lst[:dim] if len(lst) >= dim else lst + [0.0] * (dim - len(lst))

def _prep_buffer(rows, dim):
    buf = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY, alpha=BUFFER_ALPHA)
    for row in rows:
        ms = row.get("meta_state")
        if not isinstance(ms, (list, np.ndarray)):
            continue
        st = np.asarray(_pad_or_trim(ms, dim), dtype=np.float32)
        rew = float(row.get("reward", 0))
        act_raw = row.get("meta_action", {"dir": 1, "conf": 0.5})
        act = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5))) \
              if isinstance(act_raw, dict) else (int(act_raw), 0.5)
        buf.add(st, act, rew, st, True)
    logger.info("Replay buffer populated: %d samples", len(buf))
    return buf

def _append_csv(epoch_idx, avg_r):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)   # NEW 🟢
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([epoch_idx, avg_r])

# ╭──────────────────────────────────────────────────────────╮
# │ Main training loop                                      │
# ╰──────────────────────────────────────────────────────────╯
def train():
    logger.info("🚀 PPO meta‑agent training started")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data."); return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Cannot infer state_dim."); return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Buffer too small."); return

    agent  = PPOAgent(state_dim=state_dim)
    beta   = BUFFER_BETA_START
    hist   = []

    for ep in range(1, EPOCHS + 1):
        ep_rewards = []

        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights = buffer.sample(BATCH_SIZE, beta)

            states, dirs, confs, rewards, dones = [], [], [], [], []
            for idx, (s, a, r, _, done) in enumerate(batch):
                s_vec = _pad_or_trim(s, state_dim)
                states.append(s_vec)

                if isinstance(a, (list, tuple)):
                    dirs.append(int(a[0])); confs.append(float(a[1]))
                else:
                    dirs.append(int(a));    confs.append(0.5)
                rewards.append(float(r));  dones.append(int(bool(done)))

            if not states:
                continue

            # numpy → tensor
            states_t  = torch.tensor(np.array(states, dtype=np.float32))
            next_t    = states_t.clone()  # 1‑step bootstrap
            dirs_t    = torch.tensor(np.array(dirs, dtype=np.int64))
            confs_t   = torch.tensor(np.array(confs, dtype=np.float32))
            weights_t = torch.tensor(np.array(weights, dtype=np.float32))
            old_logp  = torch.zeros(len(states_t))

            td_err = agent.train_step(
                states_t, dirs_t, confs_t,
                rewards=rewards,
                dones=dones,
                next_states=next_t,
                old_logp=old_logp,
                weights=weights_t
            )

            buffer.update_priorities(idxs, td_err.cpu().tolist())
            ep_rewards.extend(rewards)

        avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        hist.append(avg_r)
        _append_csv(ep, avg_r)
        logger.info("Epoch %d/%d – avg_reward = %.4f", ep, EPOCHS, avg_r)

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep,
                 "avg_reward": avg_r,
                 "reward_std": float(np.std(ep_rewards)) if ep_rewards else 0},
                hist
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

if __name__ == "__main__":
    train()