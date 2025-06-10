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

CSV_PATH       = "meta/reward_history.csv"
NOTIFY_EVERY   = 10
ACTION_DIM     = 3                      # categorical actions 0/1/2

# ── helpers ────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _prep_buffer(rows):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    for cur in rows:
        st  = build_meta_state_from_log(cur)
        nxt = build_meta_state_from_log(cur)
        rew = compute_shaped_reward(cur)
        act = cur.get("meta_action", {"dir": 1, "conf": 0.5})
        buf.add(st, act, rew, nxt, True)
    return buf

def _append_csv(ep, val):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "avg_reward"])
        w.writerow([ep, val])

# ── main ───────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data found.")
        return

    # determine state_dim from first row
    sample_state = build_meta_state_from_log(rows[0])
    state_dim    = sample_state.shape[-1]
    save_meta_agent_dims(state_dim, ACTION_DIM)

    buffer = _prep_buffer(rows)
    agent  = PPOAgent(state_dim=state_dim)
    beta   = BUFFER_BETA_START

    history = []
    for ep in range(1, EPOCHS + 1):
        all_r = []
        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights = buffer.sample(BATCH_SIZE, beta)

            states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            next_s = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            actions_dir  = torch.tensor([b[1]["dir"]  for b in batch], dtype=torch.long)
            actions_conf = torch.tensor([b[1]["conf"] for b in batch], dtype=torch.float32)

            rewards = [b[2] for b in batch]
            dones   = [b[4] for b in batch]
            old_log = torch.zeros(len(batch))

            td = agent.train_step(
                states,
                actions_dir,
                actions_conf,
                rewards,
                dones,
                next_s,
                old_log,
                torch.tensor(weights, dtype=torch.float32)
            )
            buffer.update_priorities(idxs, td)
            all_r.extend(rewards)

        avg = float(np.mean(all_r))
        history.append(avg)
        _append_csv(ep, avg)
        logger.info(f"Epoch {ep}/{EPOCHS}  avg_reward={avg:.4f}")

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report(
                {"epoch": ep, "avg_reward": avg, "reward_std": float(np.std(all_r))},
                history
            )

    agent.save()
    update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

if __name__ == "__main__":
    train()