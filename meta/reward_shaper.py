# meta/reward_shaper.py

import numpy as np
from collections import deque

# Rolling window for Sharpe-style reward shaping
reward_window = deque(maxlen=20)

def compute_reward(trade, market_data, exit_reason=None):
    """
    Compute base shaped reward for the PPO meta-agent based on trade outcome and context.

    Args:
        trade (dict): Trade details (entry, exit, pnl, confidence, trade_type, etc.)
        market_data (dict): Market context at exit, e.g., indicators, VIX.
        exit_reason (str, optional): Reason for trade exit.

    Returns:
        float: Raw shaped reward (before Sharpe normalization).
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

    return reward


def compute_sharpe_style_reward(returns, risk_free_rate=0.0, epsilon=1e-8):
    """
    Compute a Sharpe-style risk-adjusted reward from a sequence of returns.

    Args:
        returns (list or np.array): Sequence of raw reward values.
        risk_free_rate (float): Risk free rate baseline (default 0).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: Sharpe ratio style reward.
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns) + epsilon
    sharpe_ratio = mean_return / std_return
    return sharpe_ratio


def compute_shaped_reward(log_entry):
    """
    Compute final risk-adjusted reward using Sharpe-style normalization.
    """
    trade = log_entry.get("trade", {})
    market_data = log_entry.get("market", {})
    exit_reason = log_entry.get("exit_reason")

    raw_reward = compute_reward(trade, market_data, exit_reason)

    # Sharpe-style normalization using rolling window
    reward_window.append(raw_reward)
    if len(reward_window) < reward_window.maxlen:
        return np.clip(raw_reward, -1.0, 1.0)  # not enough history yet

    mean = np.mean(reward_window)
    std = np.std(reward_window) + 1e-6
    sharpe_reward = (raw_reward - mean) / std

    return float(np.clip(sharpe_reward, -1.0, 1.0))