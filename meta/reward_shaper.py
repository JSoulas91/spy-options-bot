"""
Reward‑shaping utilities for the meta‑agent.
Encourages exploration and big, high-confidence trades.
"""

from __future__ import annotations
import csv, os, datetime, random
from collections import deque
from pathlib     import Path

import numpy as np
from utils.logger import bot_logger as logger

# ─────────────────────────────────────────────────────────
ROLL_WINDOW = 20
reward_window: deque[float] = deque(maxlen=ROLL_WINDOW)

HIST_CSV   = Path("meta/reward_history.csv")
HIST_CSV.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
def compute_reward(trade: dict, market_data: dict, exit_reason: str | None = None) -> float:
    """
    Build a shaped reward from raw PnL plus contextual signals.
    Encourages exploration and high-quality trades.
    """
    pnl         = float(trade.get("pnl", 0))
    confidence  = float(trade.get("confidence", 0))
    trade_type  = int(trade.get("trade_type", 0))  # 0 = Day, 1 = Swing
    reward      = pnl

    # 💡 Confidence shaping (scaled by pnl magnitude)
    if pnl > 0:
        reward += confidence * min(pnl / 5, 3.0)  # Stronger boost for high-conf profitable trades
    elif pnl < 0:
        reward -= confidence * min(abs(pnl) / 5, 2.0)  # Penalize overconfident losses harder

    # ⏰ Time decay penalty for late day‑trades
    if trade_type == 0:
        try:
            exit_hour = int(trade.get("exit_time", "15:59").split(":")[0])
            if exit_hour >= 15 and pnl < 0:
                reward -= 0.5
        except Exception:
            pass

    # 📈 Reward scaling by PnL magnitude (incentivize big trades more)
    if abs(pnl) < 5:
        reward *= 0.2
    elif abs(pnl) < 10:
        reward *= 0.6
    elif abs(pnl) < 25:
        reward *= 1.3
    else:
        reward *= 2.0

    # 📉 Market risk via VIX
    vix = float(market_data.get("vix", 15))
    if vix > 30:
        reward -= 0.4
    elif vix > 20:
        reward -= 0.2
    elif vix > 15:
        reward -= 0.1

    # 🔚 Exit reason tweaks
    match exit_reason:
        case "Contract near expiry": reward -= 0.3
        case "Meta-agent signal":    reward += 0.3
        case "Stop loss":            reward -= 0.2
        case "Take profit":          reward += 0.3
        case "Time-based exit":      reward -= 0.1

    # 🧠 Bonus if meta-agent exits a profitable trade
    if exit_reason == "Meta-agent signal" and pnl > 10:
        reward += 0.4

    # 🎲 Persistent exploration encouragement
    if random.random() < 0.15:  # increased from 10% to 15%
        reward += 0.2           # stronger nudge toward novelty

    return reward

# ─────────────────────────────────────────────────────────
def compute_sharpe_style_reward(returns, rf: float = 0.0, eps: float = 1e-8) -> float:
    """
    Basic Sharpe‑ratio style metric for a list/array of returns.
    Returns 0 if fewer than 2 points.
    """
    if len(returns) < 2:
        return 0.0
    r = np.asarray(returns, dtype=np.float32) - rf
    return float(r.mean() / (r.std() + eps))

# ─────────────────────────────────────────────────────────
def compute_shaped_reward(log_entry: dict) -> float:
    """
    Use `compute_reward` then scale via rolling Sharpe normalization.
    """
    trade        = log_entry.get("trade", {})
    market_data  = log_entry.get("market", {})
    exit_reason  = log_entry.get("exit_reason")

    raw = compute_reward(trade, market_data, exit_reason)

    reward_window.append(raw)
    if len(reward_window) < reward_window.maxlen:
        return float(np.clip(raw, -3.0, 3.0))  # allow broader signal early

    mean = np.mean(reward_window)
    std  = np.std(reward_window) + 1e-6
    sharpe_scaled = (raw - mean) / std
    return float(np.clip(sharpe_scaled, -3.0, 3.0))  # richer feedback for bigger trades

# ─────────────────────────────────────────────────────────
def _append_csv(timestamp: str, shaped: float, raw: float):
    """
    Append reward row to CSV: timestamp, raw, shaped
    """
    is_new = not HIST_CSV.exists()
    with HIST_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "raw_reward", "shaped_reward"])
        w.writerow([timestamp, raw, shaped])

def log_reward_trend(entry: dict):
    """
    Log reward info and update CSV + rolling stats.
    Expects keys: meta_state, meta_action, shaped_reward, sharpe_reward
    """
    try:
        ts      = datetime.datetime.utcnow().isoformat()
        shaped  = entry.get("shaped_reward", 0.0)
        raw     = entry.get("sharpe_reward", 0.0)
        _append_csv(ts, raw, shaped)

        if len(reward_window) == reward_window.maxlen:
            rolling_sharpe = compute_sharpe_style_reward(list(reward_window))
            logger.info(f"[RewardTrend] windowSharpe={rolling_sharpe:.2f}  last={shaped:+.3f}")

    except Exception as exc:
        logger.warning(f"[RewardTrend] logging failed: {exc}")