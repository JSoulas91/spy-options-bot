# reward_shaper.py

from __future__ import annotations
import math
import numpy as np

from utils.logger import bot_logger as logger


class RewardShaper:
    def __init__(self, debug: bool = False):
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
        pnl = trade_result.get("pnl", 0.0)
        duration = trade_result.get("duration", 1)
        was_successful = trade_result.get("was_successful", False)

        confidence = classifier_output.get("confidence", 0.5)
        entropy = classifier_output.get("entropy", 0.0)
        prob_success = classifier_output.get("prob_success", 0.5)

        # Base reward
        base_reward = np.tanh(pnl / 30.0)
        duration_penalty = -0.015 * math.log1p(duration)
        reward = base_reward + duration_penalty

        # Classifier shaping
        confidence_bonus = (confidence - 0.5) * 1.0
        entropy_penalty = -entropy * 0.4
        classifier_shaping = confidence_bonus + entropy_penalty

        # Regime shaping
        regime_bonus = 0.0
        if regime == "bull":
            regime_bonus += 0.1
        elif regime == "bear":
            regime_bonus -= 0.1

        # Streak shaping
        streak_bonus = 0.0
        if was_successful:
            self.win_streak += 1
            self.loss_streak = 0
            streak_bonus += min(self.win_streak, 3) * 0.3
        else:
            self.loss_streak += 1
            self.win_streak = 0
            streak_bonus -= min(self.loss_streak, 3) * 0.3

        # Sharpe shaping
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
                sharpe_boost = 0.4 * (0.5 - sharpe)

        # --- Advanced Additions ---

        # Risk penalty via drawdown
        risk_penalty = -0.5 * (trade_result.get("max_drawdown", 0.0) / (abs(pnl) + 1e-6))

        # Entry quality bonus
        entry_quality = trade_result.get("entry_quality", 0.5)
        entry_timing_bonus = (entry_quality - 0.5) * 1.5

        # Risk-Reward ratio bonus
        rrr = trade_result.get("risk_reward_ratio", 1.0)
        rrr_bonus = np.tanh((rrr - 1.0)) * 0.8

        # Confidence alignment penalty
        confidence_alignment_penalty = 0.0
        if confidence < 0.55 and was_successful:
            confidence_alignment_penalty = -0.3
        elif confidence > 0.7 and not was_successful:
            confidence_alignment_penalty = -0.5

        # Trade frequency shaping
        trades_today = trade_result.get("trades_today", 0)
        trade_count_bonus = 0.2 if 2 <= trades_today <= 6 else -0.2 if trades_today == 0 or trades_today > 8 else 0.0

        # Missed opportunity shaping
        missed_opportunity_penalty = -0.5 if trade_result.get("skipped_strong_signal", False) else 0.0

        # Directional accuracy
        direction_bonus = 0.4 if trade_result.get("direction_correct", None) is True else -0.4 if trade_result.get("direction_correct", None) is False else 0.0

        # Time-to-target shaping
        speed_bonus = 0.5 * math.exp(-trade_result.get("time_to_target", 30) / 20)

        # Final reward calculation
        total_reward = (
            reward + classifier_shaping + regime_bonus + streak_bonus +
            sharpe_boost + risk_penalty + entry_timing_bonus + rrr_bonus +
            confidence_alignment_penalty + trade_count_bonus + missed_opportunity_penalty +
            direction_bonus + speed_bonus
        )

        # Clipping
        total_reward = max(min(total_reward, 6), -6)

        # Logging
        if abs(total_reward) > 4.5:
            logger.info(f"⚡ High reward: {total_reward:.2f} → "
                        f"PNL={pnl:.2f}, conf={confidence:.2f}, rrr={rrr:.2f}, drawdown={trade_result.get('max_drawdown', 0):.2f}, "
                        f"streak_bonus={streak_bonus:.2f}, entry_quality={entry_quality:.2f}, "
                        f"trades_today={trades_today}, dir_correct={trade_result.get('direction_correct', '?')}")

        if self.debug:
            logger.info(f"🔎 Reward components:")
            logger.info(f"  base_reward={base_reward:.4f}, duration_penalty={duration_penalty:.4f}")
            logger.info(f"  confidence_bonus={confidence_bonus:.4f}, entropy_penalty={entropy_penalty:.4f}")
            logger.info(f"  regime_bonus={regime_bonus:.4f}, streak_bonus={streak_bonus:.4f}")
            logger.info(f"  sharpe_boost={sharpe_boost:.4f}, risk_penalty={risk_penalty:.4f}")
            logger.info(f"  entry_timing_bonus={entry_timing_bonus:.4f}, rrr_bonus={rrr_bonus:.4f}")
            logger.info(f"  confidence_alignment_penalty={confidence_alignment_penalty:.4f}")
            logger.info(f"  trade_count_bonus={trade_count_bonus:.4f}, missed_opportunity_penalty={missed_opportunity_penalty:.4f}")
            logger.info(f"  direction_bonus={direction_bonus:.4f}, speed_bonus={speed_bonus:.4f}")
            logger.info(f"  total_reward={total_reward:.4f}")

        return total_reward