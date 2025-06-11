# meta/train_meta_agent.py
import os, sys, json, csv
from typing import List, Dict

import numpy as np
import torch

# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meta.ppo                   import PPOAgent
from meta.meta_state            import build_meta_state_from_log
from meta.reward_shaper         import compute_shaped_reward
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

CSV_PATH     = "meta/reward_history.csv"
NOTIFY_EVERY = 10
ACTION_DIM   = 3   # categorical actions 0/1/2

# ── helpers ────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _discover_state_dim(rows: List[Dict]) -> int:
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list) and len(ms) > 0:
            return len(ms)
    return -1

def _prep_buffer(rows, state_dim: int):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    skipped = 0
    for cur in rows:
        ms = cur.get("meta_state")
        if not ms or len(ms) != state_dim:
            skipped += 1
            continue
        st  = np.asarray(ms, dtype=np.float32)
        nxt = st
        rew = float(cur.get("reward", 0))

        a = cur.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(a, dict):
            act = (int(a.get("dir", 1)), float(a.get("conf", 0.5)))
        else:
            # fallback for scalar / numpy scalar
            try:
                act = (int(a), 0.5)
            except Exception:
                act = (int(float(a)), 0.5)

        buf.add(st, act, rew, nxt, True, error=1.0)
    logger.info(f"Prepared buffer: kept {len(buf)} rows, skipped {skipped}.")
    return buf

def _append_csv(epoch_idx: int, avg_reward: float):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([epoch_idx, avg_reward])

# ── main ───────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data found.")
        return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Unable to infer consistent state_dim — aborting.")
        return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Not enough compatible samples in buffer — aborting.")
        return

    agent = PPOAgent(state_dim=state_dim)
    beta  = BUFFER_BETA_START
    history = []

    for ep in range(1, EPOCHS + 1):
        all_rewards = []

        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights, *_ = buffer.sample(BATCH_SIZE, beta)

            states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            next_s = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            dir_list, conf_list = [], []
            for a in (b[1] for b in batch):
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    dir_list.append(int(a[0]))
                    conf_list.append(float(a[1]))
                else:
                    try:
                        dir_list.append(int(a))
                    except Exception:
                        dir_list.append(int(float(a)))
                    conf_list.append(0.5)

            actions_dir  = torch.tensor(dir_list,  dtype=torch.long)
            actions_conf = torch.tensor(conf_list, dtype=torch.float32)

            rewards = [b[2] for b in batch]
            dones   = [b[4] for b in batch]
            old_log = torch.zeros(len(batch))

            td_err = agent.train_step(
                states, actions_dir, actions_conf,
                rewards, dones, next_s, old_log,
                torch.tensor(weights, dtype=torch.float32)
            )
            buffer.update_priorities(idxs, td_err)
            all_rewards.extend(rewards)

        avg_r = float(np.mean(all_rewards))
        history.append(avg_r)
        _append_csv(ep, avg_r)
        logger.info(f"Epoch {ep}/{EPOCHS}  avg_reward={avg_r:.4f}")

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep, "avg_reward": avg_r,
                 "reward_std": float(np.std(all_rewards))},
                history
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

if __name__ == "__main__":
    train()