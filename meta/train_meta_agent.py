# meta/train_meta_agent.py
import os, sys, json, csv
from typing import List, Dict

import numpy as np
import torch

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

CSV_PATH, NOTIFY_EVERY, ACTION_DIM = "meta/reward_history.csv", 10, 3

# ────────────────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH):
        return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _discover_state_dim(rows: List[Dict]) -> int:
    for r in rows:
        ms = r.get("meta_state")
        if isinstance(ms, list) and len(ms) > 1:   # ignore length‑1 scalars
            return len(ms)
    return -1

def _pad_or_trim(vec: List[float], dim: int) -> np.ndarray:
    """Ensure meta_state has exact length dim."""
    if len(vec) >= dim:
        return np.asarray(vec[:dim], dtype=np.float32)
    # shorter: pad zeros
    return np.asarray(vec + [0.0] * (dim - len(vec)), dtype=np.float32)

def _prep_buffer(rows, state_dim: int):
    buf, skipped = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA), 0
    for cur in rows:
        ms = cur.get("meta_state")
        if not isinstance(ms, list):
            skipped += 1
            continue
        st  = _pad_or_trim(ms, state_dim)
        nxt = st
        rew = float(cur.get("reward", 0))

        a = cur.get("meta_action", {"dir": 1, "conf": 0.5})
        if isinstance(a, dict):
            act = (int(a.get("dir", 1)), float(a.get("conf", 0.5)))
        else:
            act = (int(float(a)), 0.5)

        buf.add(st, act, rew, nxt, True, error=1.0)
    logger.info(f"Prepared buffer: kept {len(buf)} rows, skipped {skipped}.")
    return buf

def _append_csv(ep, val):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerows([["epoch", "avg_reward"]] if new else [], [[ep, val]])

# ── main ───────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training")
    update_status("last_ppo_attempt")

    rows = _load_rows()
    if not rows:
        logger.warning("No training data found."); return

    state_dim = _discover_state_dim(rows)
    if state_dim <= 0:
        logger.error("Cannot infer valid state_dim. Abort."); return

    save_meta_agent_dims(state_dim, ACTION_DIM)
    buffer = _prep_buffer(rows, state_dim)
    if len(buffer) < BATCH_SIZE:
        logger.error("Buffer too small after filtering. Abort."); return

    agent, beta, history = PPOAgent(state_dim=state_dim), BUFFER_BETA_START, []

    for ep in range(1, EPOCHS + 1):
        all_r = []
        for _ in range(max(1, len(buffer) // BATCH_SIZE)):
            batch, idxs, weights, *_ = buffer.sample(BATCH_SIZE, beta)

            states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            next_s = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            dirs, confs = [], []
            for a in (b[1] for b in batch):
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    dirs.append(int(a[0])); confs.append(float(a[1]))
                else:
                    dirs.append(int(float(a))); confs.append(0.5)

            td_err = agent.train_step(
                states,
                torch.tensor(dirs,  dtype=torch.long),
                torch.tensor(confs, dtype=torch.float32),
                [b[2] for b in batch],
                [b[4] for b in batch],
                next_s,
                torch.zeros(len(batch)),
                torch.tensor(weights, dtype=torch.float32)
            )
            buffer.update_priorities(idxs, td_err)
            all_r.extend([b[2] for b in batch])

        avg = float(np.mean(all_r)); history.append(avg); _append_csv(ep, avg)
        logger.info(f"Epoch {ep}/{EPOCHS}  avg_reward={avg:.4f}")

        beta = min(1.0, beta + BUFFER_BETA_INCREMENT)
        if ep % NOTIFY_EVERY == 0:
            send_training_report({"epoch": ep, "avg_reward": avg,
                                  "reward_std": float(np.std(all_r))}, history)

    agent.save(); update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training completed.")

if __name__ == "__main__":
    train()