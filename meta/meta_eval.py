# meta/meta_eval.py

import torch
import random
import numpy as np
from meta.ppo import PPOAgent
from meta.meta_state import build_meta_state_for_entry
from utils.vix import get_vix_level  # Or mock it if unavailable

# Load the trained PPO meta-agent
model_path = "meta/saved_model/ppo_meta_agent.pth"
meta_agent = PPOAgent(state_dim=11, action_dim=2)
meta_agent.load(model_path)

# Simulate synthetic scenarios
def generate_synthetic_market_data():
    price = round(random.uniform(420, 460), 2)
    ema_20 = price + random.uniform(-2, 2)
    rsi = random.uniform(10, 90)
    macd = random.uniform(-3, 3)
    volume = random.randint(5_000_000, 20_000_000)
    return {
        "price": price,
        "ema_20": ema_20,
        "rsi": rsi,
        "macd": macd,
        "volume": volume
    }

# Evaluation loop
def evaluate_policy(meta_agent, num_samples=100):
    actions = []
    vix_vals = []
    rsis = []
    confidences = []

    for _ in range(num_samples):
        market_data = generate_synthetic_market_data()
        confidence_score = round(random.uniform(0.3, 0.99), 2)
        trade_type = random.choice([0, 1])  # 0 = day, 1 = swing
        vix = get_vix_level() if callable(get_vix_level) else random.uniform(12, 35)

        meta_state = build_meta_state_for_entry(
            market_data, confidence_score, trade_type, vix_level=vix
        )

        action = meta_agent.select_action(meta_state)
        actions.append(action)
        rsis.append(market_data["rsi"])
        vix_vals.append(vix)
        confidences.append(confidence_score)

    # Results summary
    action_counts = {0: actions.count(0), 1: actions.count(1)}
    print("Meta-Agent Policy Evaluation Summary")
    print("===================================")
    print(f"Total Samples: {num_samples}")
    print(f"Do Not Trade Actions: {action_counts[0]}")
    print(f"Trade Actions: {action_counts[1]}")
    print(f"Trade %: {round(100 * action_counts[1] / num_samples, 2)}%")
    print(f"Avg RSI: {round(np.mean(rsis), 2)}")
    print(f"Avg Confidence: {round(np.mean(confidences), 2)}")
    print(f"Avg VIX: {round(np.mean(vix_vals), 2)}")

if __name__ == "__main__":
    evaluate_policy(meta_agent)