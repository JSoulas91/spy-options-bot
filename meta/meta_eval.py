# meta/meta_eval.py
"""
Quick, synthetic evaluation of the PPO meta‑agent.

• Generates random‑but‑plausible indicator snapshots
• Builds proper entry meta‑states (matching the live code)
• Runs the trained agent and prints action distribution
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch

from meta.ppo              import PPOAgent
from meta.meta_state       import build_meta_state_for_entry
from utils.vix_utils       import get_current_vix           # ✅ correct import
from config                import META_MODEL_PATH

# ─────────────────────────────────────────────
# Helper – create a one‑row DataFrame that looks like live MTF rows
# ─────────────────────────────────────────────
def _df_from_dict(d: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([d])
    # make sure all numeric cols are the right dtype
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _synthetic_tf_sample() -> Dict[str, Any]:
    """Produce one synthetic indicator row."""
    price   = round(random.uniform(420, 460), 2)
    return {
        "price": price,
        "ema_20": price + random.uniform(-2, 2),
        "volume": random.randint(5_000_000, 20_000_000),
        "rsi": random.uniform(10, 90),
        "macd": random.uniform(-3, 3),
    }

# ─────────────────────────────────────────────
def generate_synthetic_state() -> Tuple[np.ndarray, float]:
    """
    Builds a complete meta‑state (entry) using five dummy TF dataframes
    and returns (state_vector, confidence_score).
    """
    # five independent TF snapshots
    tf_1m   = _df_from_dict(_synthetic_tf_sample())
    tf_5m   = _df_from_dict(_synthetic_tf_sample())
    tf_15m  = _df_from_dict(_synthetic_tf_sample())
    tf_1h   = _df_from_dict(_synthetic_tf_sample())
    tf_1d   = _df_from_dict(_synthetic_tf_sample())

    confidence = round(random.uniform(0.3, 0.99), 2)
    trade_type = random.choice([0, 1])            # 0 = day, 1 = swing

    state_vec = build_meta_state_for_entry(
        tf_1m, tf_5m, tf_15m, tf_1h, tf_1d,
        confidence_score=confidence,
        trade_type=trade_type,
        past_trades=[],                     # no history
        long_term_data={},                  # empty LT cache
        position_size=0.0,
    )
    return state_vec, confidence

# ─────────────────────────────────────────────
def load_agent(state_dim: int) -> PPOAgent:
    """Instantiate PPOAgent with correct dim and load checkpoint if present."""
    agent = PPOAgent(state_dim=state_dim)
    ckpt = Path(META_MODEL_PATH)
    if ckpt.is_file():
        agent._load()                       # uses internal load routine
    else:
        print(f"⚠️  No checkpoint at {META_MODEL_PATH} – using fresh weights.")
    agent.net.eval()
    return agent

# ─────────────────────────────────────────────
def evaluate_policy(num_samples: int = 200):
    # Derive state_dim automatically
    sample_state, _ = generate_synthetic_state()
    state_dim = len(sample_state)

    agent       = load_agent(state_dim)
    action_hist = []           # 0 = veto, 1 = neutral, 2 = tighten‑exit
    rsis        = []
    vix_vals    = []
    confidences = []

    for _ in range(num_samples):
        state_vec, conf = generate_synthetic_state()
        vix             = get_current_vix() or random.uniform(12, 35)

        # PPOAgent.act expects a tensor
        tensor_state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        act_idx, agent_conf, *_ = agent.act(tensor_state)

        action_hist.append(act_idx)
        rsis.append(state_vec[3])           # rough ‑‑ RSI sits early in vector
        vix_vals.append(vix)
        confidences.append(conf)

    # ── summary ────────────────────────────
    veto   = action_hist.count(0)
    trade  = action_hist.count(1) + action_hist.count(2)
    print("\nMeta‑Agent Synthetic Evaluation")
    print("────────────────────────────────")
    print(f"Samples analysed    : {num_samples}")
    print(f"Veto / skip actions : {veto}")
    print(f"Trade‑ish actions   : {trade}")
    print(f"Trade %             : {trade / num_samples * 100:.2f}%")
    print(f"Avg synthetic RSI   : {np.mean(rsis):.2f}")
    print(f"Avg confidence      : {np.mean(confidences):.2f}")
    print(f"Avg VIX (used)      : {np.mean(vix_vals):.2f}\n")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_policy()