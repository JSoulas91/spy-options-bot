# meta/train_meta_agent.py
"""
Offline training script for the dual‑head PPO meta‑agent.

• Loads jsonl trades from META_LOG_PATH
• Builds a PrioritizedExperienceReplay buffer
• Trains for EPOCHS epochs, batch size BATCH_SIZE
• Saves model + basic reward‑history CSV
"""

import os, sys, json, csv
from typing import List, Dict

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo                    import PPOAgent
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

CSV_PATH         = "meta/reward_history.csv"
NOTIFY_EVERY     = 10
BUFFER_CAPACITY  = 10_000          # can tweak

# ╭──────────────────────────────────────────────────╮
# │ Helper functions                                │
# ╰──────────────────────────────────────────────────╯
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        logger.error("No meta log found: %s", META_LOG_PATH)
        return []
    with open(META_LOG_PATH) as fh:
        return [json.loads(l) for l in fh if l.strip()]

def _discover_state_dim(rows) -> int:
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list):
            return len(ms)
    return -1

def _pad_or_trim(vec, dim):
    if not isinstance(vec, (list, np.ndarray)):
        return [0.0] * dim
    lst = list(vec)
    return lst[:dim] if len(lst) >= dim else lst + [0.0] * (dim - len(lst))

def _prep_buffer(rows, state_dim):
    buf = PrioritizedReplayBuffer(capacity=BUFFER_CAPACITY, alpha=BUFFER_ALPHA)
    for row in rows:
        state_vec = row.get("meta_state")
        if not isinstance(state_vec, list):
            continue
        st = np.asarray(_pad_or_trim(state_vec, state_dim), dtype=np.float32)

        act_raw = row.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(act_raw, dict):
            action = (int(act_raw.get("dir", 1)), float(act_raw.get("conf", 0.5)))
        else:
            action = (int(act_raw), 0.5)

        reward = float(row.get("reward", 0))
        buf.add(st, action, reward, st, True)  # next_state = st (1‑step)
    logger.info("Replay buffer populated: %d samples", len(buf))
    return buf

def _append_csv(epoch_i, avg_r):
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([epoch_i, avg_r])

# ╭──────────────────────────────────────────────────╮
# │ Main training routine                            │
# ╰──────────────────────────────────────────────────╯
def train():
    logger.info("🚀 PPO meta‑agent training started")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data – aborting."); return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Failed to infer state dimension."); return
    save_meta_agent_dims(state_dim, 3)

    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Buffer too small (%d) for batch size %d", len(buffer), BATCH_SIZE)
        return

    agent = PPOAgent(state_dim=state_dim)
    beta  = BUFFER_BETA_START
    reward_history = []

    for ep in range(1, EPOCHS + 1):
        ep_rs = []
        # number of mini‑batches per epoch
        n_mb = max(1, len(buffer) // BATCH_SIZE)
        for _ in range(n_mb):
            (states_np, actions_np, rewards_np, _, dones_np), idxs, w = buffer.sample(BATCH_SIZE, beta)

            # ── unpack & tensorise ─────────────────────────
            states = torch.tensor(states_np, dtype=torch.float32)
            next_s = states.clone()                        # 1‑step bootstrap
            dirs   = torch.tensor([a[0] for a in actions_np], dtype=torch.long)
            confs  = torch.tensor([a[1] for a in actions_np], dtype=torch.float32)
            rewards= rewards_np.tolist()
            dones  = dones_np.tolist()
            weights= torch.tensor(w, dtype=torch.float32)
            old_lp = torch.zeros(len(states))

            td_err = agent.train_step(
                states, dirs, confs, rewards, dones,
                next_s, old_lp, weights
            )
            buffer.update_priorities(idxs, td_err.cpu().numpy().tolist())
            ep_rs.extend(rewards)

        avg_r = float(np.mean(ep_rs)) if ep_rs else 0.0
        reward_history.append(avg_r)
        _append_csv(ep, avg_r)
        logger.info("Epoch %d/%d | Avg reward %.4f | β=%.3f",
                    ep, EPOCHS, avg_r, beta)

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep, "avg_reward": avg_r,
                 "reward_std": float(np.std(ep_rs)) if ep_rs else 0.0},
                reward_history
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ PPO meta‑agent training completed.")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()