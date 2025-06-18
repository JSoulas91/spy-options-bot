"""
Reward‑shaping utilities for the meta‑agent.
Encourages exploration, clean execution, and high-confidence trades.
"""

from __future__ import annotations
import csv, os, datetime, random
from collections import deque
from pathlib import Path

import numpy as np
from utils.logger import bot_logger as logger

# ─────────────────────────────────────────────────────────
ROLL_WINDOW = 20
MIN_HIGH_REWARD = 1.5  # New: Used to help prioritize high-quality experiences
reward_window: deque[float] = deque(maxlen=ROLL_WINDOW)

HIST_CSV = Path("meta/reward_history.csv")
HIST_CSV.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
def compute_reward(trade: dict, market_data: dict, exit_reason: str | None = None) -> float:
    """
    Build a shaped reward from raw PnL plus contextual signals.
    Encourages exploration and high-quality trades.
    """
    pnl           = float(trade.get("pnl", 0))
    confidence    = float(trade.get("confidence", 0))
    trade_type    = int(trade.get("trade_type", 0))  # 0 = Day, 1 = Swing
    setup_quality = float(trade.get("setup_quality", 0.0))
    num_signals   = int(trade.get("total_signals_today", 10))
    realized_vol  = float(market_data.get("realized_vol", 1.0))
    vix           = float(market_data.get("vix", 15))
    entry_time    = trade.get("entry_time", "09:35")
    exit_time     = trade.get("exit_time", "15:59")
    post_exit_move = float(trade.get("post_exit_move", 0.0))

    reward = np.sign(pnl) * np.log1p(abs(pnl))

    # 💡 Confidence shaping
    reward += np.clip(confidence, 0, 1) * np.sign(pnl) * min(abs(pnl) / 5, 3.0)

    # ⏰ Late exit penalty for day-trades
    if trade_type == 0:
        try:
            exit_hour = int(exit_time.split(":")[0])
            if exit_hour >= 15 and pnl < 0:
                reward -= 0.5
        except:
            pass

    # 📉 Risk environment penalty via VIX
    reward -= 0.1 * max(0, (vix - 15) // 5)

    # 🎯 Smarter exploration bonus — only if uncertain and not in extreme regimes
    if 0.4 < confidence < 0.7 and realized_vol < 2.0 and random.random() < 0.2:
        reward += 0.25

    # 🔚 Exit reason shaping
    match exit_reason:
        case "Contract near expiry": reward -= 0.3
        case "Meta-agent signal":    reward += 0.3
        case "Stop loss":            reward -= 0.2
        case "Take profit":          reward += 0.3
        case "Time-based exit":      reward -= 0.1

    if exit_reason == "Meta-agent signal" and pnl > 10:
        reward += 0.4

    # 🧠 Setup quality shaping
    reward += np.clip(setup_quality, 0, 1) * 0.5

    # 📊 Selectivity bonus
    if confidence > 0.8 and num_signals <= 3:
        reward += 0.4

    # ⚡ Fast win bonus
    try:
        entry_hour = int(entry_time.split(":")[0])
        exit_hour  = int(exit_time.split(":")[0])
        if pnl > 10 and (exit_hour - entry_hour) <= 2:
            reward += 0.5
    except:
        pass

    # 🌀 Fluke discouragement
    if confidence < 0.4 and pnl > 10:
        reward *= 0.6

    # 🧠 Classifier agreement/disagreement
    if setup_quality > 0.7 and pnl > 10:
        reward += 0.4
    elif setup_quality < 0.3 and pnl < 0:
        reward += 0.2

    if confidence > 0.85 and setup_quality > 0.8:
        reward += 0.4

    # 📈 PnL scaling tiers (adjusted)
    if abs(pnl) < 5:
        reward *= 0.7 if confidence > 0.6 else 0.3  # Still positive, just modest
    elif abs(pnl) < 10:
        reward *= 0.9
    elif abs(pnl) < 25:
        reward *= 1.2
    else:
        reward *= 1.6

    # 📉 Realized volatility shaping
    reward *= max(0.8, 1.5 - realized_vol)

    # 🏆 Rare excellent trade multiplier
    if confidence > 0.9 and setup_quality > 0.8 and pnl > 20:
        reward *= 1.5

    # ❌ Missed post-trade opportunity penalty
    if pnl > 5 and post_exit_move > 10:
        reward -= 0.4

    # ❌ Classifier disagreement penalty
    if setup_quality < 0.3 and confidence > 0.8 and pnl < 0:
        reward -= 0.5

    # 💰 Extra large win bonus
    if abs(pnl) > 30:
        reward += 0.5

    # 📉 Adaptive exploration if rolling Sharpe is weak
    if len(reward_window) == reward_window.maxlen:
        rolling_sharpe = compute_sharpe_style_reward(list(reward_window))
        if rolling_sharpe < 0.1 and random.random() < 0.2:
            reward += 0.3

    return float(reward)

# ─────────────────────────────────────────────────────────
def compute_sharpe_style_reward(returns, rf: float = 0.0, eps: float = 1e-8) -> float:
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
        return float(np.clip(raw, -3.0, 3.0))

    mean = np.mean(reward_window)
    std  = np.std(reward_window) + 1e-6
    sharpe_scaled = (raw - mean) / std
    return float(np.clip(sharpe_scaled, -3.0, 3.0))

# ─────────────────────────────────────────────────────────
def _append_csv(timestamp: str, shaped: float, raw: float):
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