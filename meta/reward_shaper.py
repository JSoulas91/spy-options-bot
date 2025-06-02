# meta/reward_shaper.py

import numpy as np

def compute_reward(trade, market_data, exit_reason=None):
    """
    Compute shaped reward for the PPO meta-agent based on trade outcome and context.
    
    Args:
        trade (dict): Trade information, including entry, exit, pnl, confidence, trade_type.
        market_data (dict): Market context at exit, e.g., VIX, trend, volume, indicators.
        exit_reason (str, optional): Reason for trade exit (e.g., 'Meta-agent signal', 'DTE', 'Stop Loss').

    Returns:
        float: Shaped reward value.
    """
    pnl = float(trade.get("pnl", 0))
    confidence = float(trade.get("confidence", 0))
    trade_type = int(trade.get("trade_type", 0))  # 0 = Day, 1 = Swing

    reward = pnl

    # Adaptive reward coefficients
    confidence_weight = 0.5 if confidence >= 0.7 else 0.2
    vix = float(market_data.get("vix", 15))
    vix_penalty = 0.2 if vix > 20 else 0.1

    # 📈 Bonus for confidence-aligned wins
    if pnl > 0:
        reward += confidence_weight

    # 📉 Penalty for high-confidence losses
    if pnl < 0:
        reward -= confidence_weight

    # ⏰ Penalty for late exits in day trades
    if trade_type == 0:
        exit_hour = int(trade.get("exit_time", "15:59").split(":")[0])
        if exit_hour == 15 and pnl < 0:
            reward -= 0.3  # Time-decay penalty

    # ⚖️ Smooth penalty for flat or low PnL trades
    if abs(pnl) < 5:
        reward *= 0.25

    # ❗ Penalty for risky market context (high VIX)
    reward -= vix_penalty

    # 🧠 Penalty for override exits
    if exit_reason == "Contract near expiry":
        reward -= 0.2
    elif exit_reason == "Meta-agent signal":
        reward += 0.1  # Optional: reward if agent initiated exit

    # 🔁 Normalize reward
    reward = np.clip(re 