"""
Quick, synthetic evaluation of the PPO meta‑agent.

• Generates random‑but‑plausible indicator snapshots
• Builds proper entry meta‑states (matching the live code)
• Injects dummy classifier predictions to test full pipeline
• Runs the trained agent and prints action distribution and logits
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from meta.ppo              import PPOAgent
from meta.meta_state       import build_meta_state_for_entry
from utils.vix_utils       import get_current_vix
from config                import META_MODEL_PATH

# ─────────────────────────────────────────────
def _df_from_dict(d: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([d])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _synthetic_tf_sample() -> Dict[str, Any]:
    price = round(random.uniform(420, 460), 2)
    return {
        "price": price,
        "ema_20": price + random.uniform(-2, 2),
        "volume": random.randint(5_000_000, 20_000_000),
        "rsi": random.uniform(10, 90),
        "macd": random.uniform(-3, 3),
    }

# ─────────────────────────────────────────────
def generate_synthetic_state() -> Tuple[np.ndarray, float]:
    tf_1m  = _df_from_dict(_synthetic_tf_sample())
    tf_5m  = _df_from_dict(_synthetic_tf_sample())
    tf_15m = _df_from_dict(_synthetic_tf_sample())
    tf_1h  = _df_from_dict(_synthetic_tf_sample())
    tf_1d  = _df_from_dict(_synthetic_tf_sample())

    confidence  = round(random.uniform(0.3, 0.99), 2)
    trade_type  = random.choice([0, 1])  # 0 = day, 1 = swing

    # Inject fake classifier predictions
    classifier_output = {
        "trade_success_prob": round(random.uniform(0.1, 0.95), 3),
        "predicted_direction": random.choice([0, 1]),
        "class_probabilities": [round(random.uniform(0.1, 0.9), 3) for _ in range(2)],
        "entropy": round(random.uniform(0.01, 1.0), 3),
        "regime_class": random.randint(0, 2),
    }

    state_vec = build_meta_state_for_entry(
        tf_1m, tf_5m, tf_15m, tf_1h, tf_1d,
        confidence_score=confidence,
        trade_type=trade_type,
        past_trades=[],
        long_term_data={},
        position_size=0.0,
        classifier_output=classifier_output,
    )
    return state_vec, confidence

# ─────────────────────────────────────────────
def load_agent(state_dim: int) -> PPOAgent:
    agent = PPOAgent(state_dim=state_dim)
    ckpt = Path(META_MODEL_PATH)
    if ckpt.is_file():
        agent._load()
    else:
        print(f"⚠️  No checkpoint at {META_MODEL_PATH} – using fresh weights.")
    agent.net.eval()
    return agent

# ─────────────────────────────────────────────
def evaluate_policy(num_samples: int = 200):
    sample_state, _ = generate_synthetic_state()
    state_dim = len(sample_state)

    agent = load_agent(state_dim)
    action_hist, logits_list, rsis, vix_vals, confidences = [], [], [], [], []

    for _ in range(num_samples):
        state_vec, conf = generate_synthetic_state()
        vix = get_current_vix() or random.uniform(12, 35)

        tensor_state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        act_idx, agent_conf, logits, _ = agent.act(tensor_state)

        action_hist.append(act_idx)
        logits_list.append(logits.detach().numpy()[0])
        rsis.append(state_vec[3])
        vix_vals.append(vix)
        confidences.append(conf)

    # ── summarize ─────────────────────────────
    veto = action_hist.count(0)
    trade = action_hist.count(1) + action_hist.count(2)

    print("\nMeta‑Agent Synthetic Evaluation")
    print("────────────────────────────────")
    print(f"Samples analysed      : {num_samples}")
    print(f"Veto / skip actions   : {veto}")
    print(f"Trade‑ish actions     : {trade}")
    print(f"Trade %               : {trade / num_samples * 100:.2f}%")
    print(f"Avg synthetic RSI     : {np.mean(rsis):.2f}")
    print(f"Avg confidence        : {np.mean(confidences):.2f}")
    print(f"Avg VIX (used)        : {np.mean(vix_vals):.2f}")

    # ── Softmax probabilities ─────────────────
    logits_tensor = torch.tensor(logits_list)
    probs = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()
    avg_probs = np.mean(probs, axis=0)
    print(f"Avg Action Probabilities (softmax): {avg_probs.round(3)}")

    # ── Optional: show histogram ──────────────
    try:
        plt.hist(action_hist, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.7)
        plt.xticks([0, 1, 2], ['Veto', 'Neutral', 'Tighten'])
        plt.title("Meta-Agent Action Distribution")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"(Optional plot skipped: {e})")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_policy()