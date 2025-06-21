# Reward‑shaping utilities for the meta‑agent.
# Encourages exploration, clean execution, and high-confidence trades.

from __future__ import annotations
import csv, os, datetime, random
from collections import deque
from pathlib import Path

import numpy as np
from utils.logger import bot_logger as logger

import math

logger = logging.getLogger("RewardShaper")
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose output


class RewardShaper:
    def __init__(self, debug=False):
        self.reward_history = []
        self.max_history = 100
        self.win_streak = 0
        self.loss_streak = 0
        self.debug = debug

    def reset(self):
        self.reward_history.clear()
        self.win_streak = 0
        self.loss_streak = 0

    def compute_shaped_reward(self, trade_result: dict, classifier_output: dict, regime: str):
        """
        trade_result: {
            "pnl": float,
            "duration": int,
            "was_successful": bool
        }

        classifier_output: {
            "confidence": float,
            "entropy": float,
            "prob_success": float
        }

        regime: str
        """
        pnl = trade_result.get("pnl", 0.0)
        duration = trade_result.get("duration", 1)
        was_successful = trade_result.get("was_successful", False)

        confidence = classifier_output.get("confidence", 0.5)
        entropy = classifier_output.get("entropy", 0.0)
        prob_success = classifier_output.get("prob_success", 0.5)

        # Base reward: scaled PnL
        base_reward = np.tanh(pnl / 50.0)  # Smoother scale
        duration_penalty = -0.01 * math.log1p(duration)

        reward = base_reward + duration_penalty

        # Classifier shaping
        confidence_bonus = (confidence - 0.5) * 0.5
        entropy_penalty = -entropy * 0.2
        classifier_shaping = confidence_bonus + entropy_penalty

        # Regime shaping
        regime_bonus = 0.0
        if regime == "bull":
            regime_bonus += 0.1
        elif regime == "bear":
            regime_bonus -= 0.1

        # Streak-based shaping
        streak_bonus = 0.0
        if was_successful:
            self.win_streak += 1
            self.loss_streak = 0
            streak_bonus += min(self.win_streak, 3) * 0.2
        else:
            self.loss_streak += 1
            self.win_streak = 0
            streak_bonus -= min(self.loss_streak, 3) * 0.2

        # Sharpe-aware boosting
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)

        sharpe_boost = 0.0
        if len(self.reward_history) >= 10:
            returns = np.array(self.reward_history)
            mean_r = np.mean(returns)
            std_r = np.std(returns) + 1e-6
            sharpe = mean_r / std_r

            if sharpe < 0.5:
                sharpe_boost = 0.2 * (0.5 - sharpe)

        total_reward = reward + classifier_shaping + regime_bonus + streak_bonus + sharpe_boost

        # Stronger final scaling
        total_reward = max(min(total_reward, 5), -5)

        # Debug logging
        if self.debug:
            logger.info(f"Reward components:")
            logger.info(f"  base_reward={base_reward:.4f}, duration_penalty={duration_penalty:.4f}")
            logger.info(f"  confidence_bonus={confidence_bonus:.4f}, entropy_penalty={entropy_penalty:.4f}")
            logger.info(f"  regime_bonus={regime_bonus:.4f}, streak_bonus={streak_bonus:.4f}")
            logger.info(f"  sharpe_boost={sharpe_boost:.4f}, total_reward={total_reward:.4f}")

        return total_reward