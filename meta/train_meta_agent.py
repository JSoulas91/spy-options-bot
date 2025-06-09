# meta/train_meta_agent.py – adapted for dual‑head PPO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, csv
import numpy as np, torch
from typing import List, Dict

from meta.ppo                  import PPOAgent
from meta.meta_state           import build_meta_state_from_log
from meta.reward_shaper        import compute_shaped_reward
from meta.meta_agent_info      import save_meta_agent_dims
from meta.prioritized_buffer   import PrioritizedReplayBuffer
from utils.logger              import bot_logger as logger
from utils.meta_telegram_reporter import send_training_report
from utils.telegram_utils      import send_telegram_message
from monitor.health_check      import update_status
from config import (
    META_LOG_PATH, EPOCHS, BATCH_SIZE,
    BUFFER_ALPHA, BUFFER_BETA_START, BUFFER_BETA_INCREMENT
)

CSV_PATH   = "meta/reward_history.csv"
NOTIFY_EVERY = 10

# ── helpers ─────────────────────────────────────────────────
def _load_rows() -> List[Dict]:
    if not os.path.exists(META_LOG_PATH): return []
    with open(META_LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def _prep_buffer(rows):
    buf = PrioritizedReplayBuffer(alpha=BUFFER_ALPHA)
    for cur in rows:
        st  = build_meta_state_from_log(cur)
        nxt = build_meta_state_from_log(cur)  # bootstrap 1‑step; okay
        rew = compute_shaped_reward(cur)
        act = cur.get("meta_action", {"dir":1,"conf":0.5})
        buf.add(st, act, rew, nxt, True)
    return buf

def _append_csv(ep, val):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH,'a',newline='') as f:
        w=csv.writer(f)
        if new: w.writerow(["epoch","avg_reward"])
        w.writerow([ep,val])

# ── main ────────────────────────────────────────────────────
def train():
    logger.info("🚀 Dual‑head PPO training.")
    update_status("last_ppo_attempt")
    rows = _load_rows()
    if not rows: 
        logger.warning("No data."); return

    save_meta_agent_dims(rows[0])        # keeps dims config file updated
    buffer = _prep_buffer(rows)
    agent  = PPOAgent(); beta = BUFFER_BETA_START

    history=[]
    for ep in range(1, EPOCHS+1):
        all_r=[]
        for _ in range(len(buffer)//BATCH_SIZE):
            batch, idx, w = buffer.sample(BATCH_SIZE, beta)
            # ── build tensors
            states      = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            actions_dir  = torch.tensor([b[1]["dir"]  for b in batch], dtype=torch.long)
            actions_conf = torch.tensor([b[1]["conf"] for b in batch], dtype=torch.float32)

            rewards = [b[2] for b in batch]; dones=[b[4] for b in batch]
            old_logp= torch.zeros(len(batch))          # first training – logp≈0

            td = agent.train_step({
                "states":states,"next_states":next_states,
                "actions_dir":actions_dir,
                "actions_conf":actions_conf,
                "rewards":rewards,"dones":dones,
                "old_logp":old_logp,
                "weights":torch.tensor(w,dtype=torch.float32)
            })
            buffer.update_priorities(idx, td)
            all_r.extend(rewards)

        avg=np.mean(all_r); history.append(avg); _append_csv(ep,avg)
        logger.info(f"Epoch {ep}/{EPOCHS}  avgR={avg:.3f}")
        beta=min(1.0,beta+BUFFER_BETA_INCREMENT)
        if ep%NOTIFY_EVERY==0:
            send_training_report({"epoch":ep,"avg_reward":avg,
                                  "reward_std":float(np.std(all_r)),
                                  "sharpe":0,"reject_rate":0}, history)

    agent.save(); update_status("last_ppo")
    send_telegram_message("✅ Dual‑head PPO training done.")
if __name__=="__main__":
    train()