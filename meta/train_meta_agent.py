# meta/train_meta_agent.py
import os
import sys
import json
import csv
from typing import List, Dict

import numpy as np
import torch

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo                    import PPOAgent
from meta.meta_state             import build_meta_state_from_log
from meta.reward_shaper          import compute_shaped_reward
from meta.meta_agent_info        import save_meta_agent_dims
from meta.prioritized_buffer     import PrioritizedReplayBuffer
from utils.logger                import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report
from utils.telegram_utils        import send_telegram_message
from monitor.health_check        import update_status
from config import (
    META_LOG_PATH, EPOCHS, BATCH_SIZE,
    BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT
)

CSV_PATH       = "meta/reward_history.csv"
NOTIFY_EVERY   = 10
ACTION_DIM     = 3  # categorical actions 0/1/2

# ────────────────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _discover_state_dim(rows: List[Dict]) -> int:
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list) and len(ms) >= 5:  # ignore length‑1 legacy rows
            return len(ms)
    return -1

def _pad_or_trim(vec, dim):
    """Return list exactly length dim (pad with zeros or trim)."""
    if not isinstance(vec, list):
        vec = []
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))

def _prep_buffer(rows, dim):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    skipped = 0
    for cur in rows:
        ms = cur.get("meta_state")
        if not isinstance(ms, list):
            skipped += 1
            continue
        st = np.asarray(_pad_or_trim(ms, dim), dtype=np.float32)

        rew = float(cur.get("reward", 0))
        a_raw = cur.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(a_raw, dict):
            act = (int(a_raw.get("dir", 1)), float(a_raw.get("conf", 0.5)))
        else:
            try:
                act = (int(a_raw), 0.5)
            except Exception:
                act = (0, 0.5)

        buf.add(st, act, rew, st, True, error=1.0)
    logger.info(f"Buffer ready: kept {len(buf)} rows, skipped {skipped}.")
    return buf

def _append_csv(epoch_idx: int, avg_reward: float):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([epoch_idx, avg_reward])

# ────────────────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data found.")
        return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Could not infer valid state_dim – abort.")
        return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Buffer too small for training – abort.")
        return

    agent = PPOAgent(state_dim=state_dim)
    beta  = BUFFER_BETA_START
    history = []

    for ep in range(1, EPOCHS + 1):
        all_rewards = []

        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights, *_ = buffer.sample(BATCH_SIZE, beta)

            # Prepare mini‑batch lists
            fixed_states, next_states = [], []
            dirs, confs, rewards, dones = [], [], [], []

            for s, a, r, ns, dflag, *_ in batch:  # ignore extra fields
                vec = _pad_or_trim(list(s), state_dim)
                if len(vec) != state_dim:
                    continue  # malformed, skip
                fixed_states.append(vec)
                next_states.append(_pad_or_trim(list(ns), state_dim))

                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    dirs.append(int(a[0]))
                    confs.append(float(a[1]))
                else:
                    dirs.append(int(float(a)))
                    confs.append(0.5)

                rewards.append(r)
                dones.append(dflag)

            if not fixed_states:
                continue  # no valid samples this loop

            # Convert to tensors
            states_t  = torch.tensor(fixed_states, dtype=torch.float32)
            next_t    = torch.tensor(next_states, dtype=torch.float32)
            dirs_t    = torch.tensor(dirs,  dtype=torch.long)
            confs_t   = torch.tensor(confs, dtype=torch.float32)
            weights_t = torch.tensor(weights[:len(rewards)], dtype=torch.float32)

            td_err = agent.train_step(
                states_t,
                dirs_t,
                confs_t,
                rewards,
                dones,
                next_t,
                torch.zeros(len(rewards)),   # old_logp placeholder
                weights_t
            )
            buffer.update_priorities(idxs[:len(rewards)], td_err)
            all_rewards.extend(rewards)

        avg_r = float(np.mean(all_rewards)) if all_rewards else 0.0
        history.append(avg_r)
        _append_csv(ep, avg_r)
        logger.info(f"Epoch {ep}/{EPOCHS} – avg_reward = {avg_r:.4f}")

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep, "avg_reward": avg_r,
                 "reward_std": float(np.std(all_rewards)) if all_rewards else 0.0},
                history
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()