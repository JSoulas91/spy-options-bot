# meta/reward_shaper.py

import numpy as np

def compute_reward(trade, market_data, exit_reason=None):
    """
    Compute shaped reward for the PPO meta-agent based on trade outcome and context.

    Args:
        trade (dict): Trade details (entry, exit, pnl, confidence, trade_type, etc.)
        market_data (dict): Market context at exit, e.g., indicators, VIX.
        exit_reason (str, optional): Reason for trade exit.

    Returns:
        float: Reward value (clipped and shaped).
    """
    pnl = float(trade.get("pnl", 0))
    confidence = float(trade.get("confidence", 0))
    trade_type = int(trade.get("trade_type", 0))  # 0 = Day, 1 = Swing
    reward = pnl

    # 💡 Confidence shaping
    if pnl > 0:
        reward += 0.5 if confidence >= 0.7 else 0.2
    elif pnl < 0:
        reward -= 0.5 if confidence >= 0.7 else 0.2

    # ⏰ Time penalty for poor day trade timing
    if trade_type == 0:
        try:
            exit_hour = int(trade.get("exit_time", "15:59").split(":")[0])
            if exit_hour >= 15 and pnl < 0:
                reward -= 0.3  # Time decay penalty
        except:
            pass

    # ⚖️ Weak trades (low PnL)
    if abs(pnl) < 5:
        reward *= 0.25

    # 📉 Market risk via VIX
    vix = float(market_data.get("vix", 15))
    if vix > 30:
        reward -= 0.4
    elif vix > 20:
        reward -= 0.2
    elif vix > 15:
        reward -= 0.1

    # 🔚 Exit reasons
    if exit_reason == "Contract near expiry":
        reward -= 0.2
    elif exit_reason == "Meta-agent signal":
        reward += 0.1
    elif exit_reason == "Stop loss":
        reward -= 0.1
    elif exit_reason == "Take profit":
        reward += 0.2
    elif exit_reason == "Time-based exit":
        reward -= 0.1

    # 🔁 Normalize/clamp reward
    reward = np.clip(reward, -1.0, 1.0)
    return reward