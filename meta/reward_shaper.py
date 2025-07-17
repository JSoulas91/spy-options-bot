from __future__ import annotations
import math
import os
import csv
import numpy as np

from utils.logger import bot_logger as logger


class RewardShaper:
    def __init__(self, debug: bool = False, csv_log_path: str = "reward_shaper.csv"):
        self.reward_history = []
        self.max_history = 100
        self.win_streak = 0
        self.loss_streak = 0
        self.debug = debug
        self.csv_log_path = csv_log_path

        if self.csv_log_path and not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "pct_pnl", "base_reward", "duration_penalty", "conf", "entropy",
                    "confidence_bonus", "entropy_penalty", "regime", "regime_bonus",
                    "streak_bonus", "sharpe_boost", "drawdown_penalty", "entry_quality_bonus",
                    "rrr_bonus", "confidence_alignment", "setup_bonus", "exploration_bonus",
                    "trade_count_bonus", "missed_opportunity_penalty", "direction_bonus", "speed_bonus",
                    "agent_confidence", "agent_conf_penalty", "agent_classifier_agreement",
                    "total_reward"
                ])

    def reset(self):
        self.reward_history.clear()
        self.win_streak = 0
        self.loss_streak = 0

    def compute_shaped_reward(
        self,
        trade_result: dict,
        classifier_output: dict,
        regime: str,
        agent_confidence: float = 0.5
    ):
        # Handle skipped trades
        if trade_result.get("skipped_trade", False):
            confidence = classifier_output.get("confidence", 0.5)
            setup_quality = trade_result.get("setup_quality", 0.5)

            if confidence > 0.6 and setup_quality > 0.6:
                penalty = -0.2 - (confidence - 0.6) * 0.5
            else:
                penalty = -0.2

            # Encourage exploration when PPO lacks confidence but classifier is strong
            if agent_confidence < 0.4 and confidence > 0.65:
                exploration_reward = 0.3
                if self.debug:
                    logger.info(f"🧪 Exploration bonus: PPO conf={agent_confidence:.2f}, classifier conf={confidence:.2f}")
                return exploration_reward

            if self.debug:
                logger.info(f"🚫 Skip penalty: {penalty:.3f}, conf={confidence:.2f}, setup={setup_quality:.2f}")
            return penalty

        # Standard trade shaping
        pct_pnl = trade_result.get("pct_pnl", 0.0)
        duration = trade_result.get("duration", 1)
        was_successful = trade_result.get("was_successful", False)

        assert isinstance(pct_pnl, (float, int)), f"pct_pnl is not a float/int: {pct_pnl} ({type(pct_pnl)})"
        assert -1000 < pct_pnl < 1000, f"Unrealistic pct_pnl={pct_pnl}, possible bug in trade_result"

        abs_pnl = abs(pct_pnl)
        bonus_scaler = min(1.0, abs_pnl / 10.0)

        confidence = classifier_output.get("confidence", 0.5)
        entropy = classifier_output.get("entropy", 0.0)

        base_reward = np.tanh(pct_pnl / 30.0)
        duration_penalty = -0.015 * math.log1p(duration)
        reward = base_reward + duration_penalty

        if pct_pnl < -20:
            logger.warning(f"⚠️ Severe loss: pct_pnl={pct_pnl:.2f} → Applying strong penalty")
            reward -= (abs(pct_pnl) / 15.0)  # was /20

        # Classifier shaping
        confidence_bonus = (confidence - 0.5) * 2.5 if was_successful and confidence > 0.55 else 0.0
        entropy_penalty = -entropy * 0.4
        classifier_shaping = confidence_bonus + entropy_penalty

        regime_bonus = 0.1 if regime == "bull" else -0.1 if regime == "bear" else 0.0

        # Streak
        streak_bonus = 0.0
        if was_successful:
            self.win_streak += 1
            self.loss_streak = 0
            streak_bonus += min(self.win_streak, 3) * 0.3
        else:
            self.loss_streak += 1
            self.win_streak = 0
            streak_bonus -= min(self.loss_streak, 3) * 0.2

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
                sharpe_boost = 0.8 * (0.5 - sharpe)

        drawdown = trade_result.get("max_drawdown", 0.0)
        risk_penalty = -0.5 * (drawdown / (abs(pct_pnl) + 5.0))

        entry_timing_bonus = (trade_result.get("entry_quality", 0.5) - 0.5) * 2.0  # was 1.5
        rrr = trade_result.get("risk_reward_ratio", 1.0)
        rrr_bonus = np.tanh(rrr - 1.0) * 0.8

        confidence_alignment_penalty = 0.0
        if confidence < 0.55 and was_successful:
            confidence_alignment_penalty = -0.3
        elif confidence > 0.7 and not was_successful:
            confidence_alignment_penalty = -0.5

        setup_bonus = (trade_result.get("setup_quality", 0.5) - 0.5) * 1.0
        exploration_bonus = trade_result.get("exploration_bonus", 0.0)
        trades_today = trade_result.get("trades_today", 0)
        trade_count_bonus = 0.4 if 2 <= trades_today <= 6 else -0.2 if trades_today == 0 or trades_today > 8 else 0.0
        missed_opportunity_penalty = -0.5 if trade_result.get("skipped_strong_signal", False) else 0.0
        direction_bonus = 0.4 if trade_result.get("direction_correct", None) is True else -0.4 if trade_result.get("direction_correct", None) is False else 0.0
        speed_bonus = 0.5 * math.exp(-trade_result.get("time_to_target", 30) / 20)

        # Agent shaping
        agent_conf_penalty = -0.6 if agent_confidence > 0.8 and not was_successful else 0.3 if agent_confidence > 0.8 and was_successful else 0.0
        agent_classifier_agreement = 0.4 if agent_confidence > 0.75 and confidence > 0.75 and was_successful else -0.4 if agent_confidence > 0.75 and confidence > 0.75 else 0.0

        bonus_total = bonus_scaler * (
            classifier_shaping + regime_bonus + streak_bonus + sharpe_boost +
            risk_penalty + entry_timing_bonus + rrr_bonus + confidence_alignment_penalty +
            setup_bonus + exploration_bonus + trade_count_bonus +
            missed_opportunity_penalty + direction_bonus + speed_bonus +
            agent_conf_penalty + agent_classifier_agreement
        )

        total_reward = reward + bonus_total

        if self.csv_log_path:
            with open(self.csv_log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    pct_pnl, base_reward, duration_penalty, confidence, entropy,
                    confidence_bonus, entropy_penalty, regime, regime_bonus,
                    streak_bonus, sharpe_boost, risk_penalty, entry_timing_bonus,
                    rrr_bonus, confidence_alignment_penalty, setup_bonus, exploration_bonus,
                    trade_count_bonus, missed_opportunity_penalty, direction_bonus, speed_bonus,
                    agent_confidence, agent_conf_penalty, agent_classifier_agreement,
                    total_reward
                ])

        if abs(total_reward) > 4.5:
            logger.info(f"⚡ High reward: {total_reward:.2f} → PCT_PNL={pct_pnl:.2f}, conf={confidence:.2f}, agent_conf={agent_confidence:.2f}, rrr={rrr:.2f}")

        if self.debug:
            logger.info(f"🔎 Reward breakdown:")
            logger.info(f"  base_reward={base_reward:.4f}, duration_penalty={duration_penalty:.4f}")
            logger.info(f"  confidence_bonus={confidence_bonus:.4f}, entropy_penalty={entropy_penalty:.4f}")
            logger.info(f"  regime_bonus={regime_bonus:.4f}, streak_bonus={streak_bonus:.4f}")
            logger.info(f"  sharpe_boost={sharpe_boost:.4f}, risk_penalty={risk_penalty:.4f}")
            logger.info(f"  entry_timing_bonus={entry_timing_bonus:.4f}, rrr_bonus={rrr_bonus:.4f}")
            logger.info(f"  confidence_alignment_penalty={confidence_alignment_penalty:.4f}")
            logger.info(f"  setup_bonus={setup_bonus:.4f}, exploration_bonus={exploration_bonus:.4f}")
            logger.info(f"  trade_count_bonus={trade_count_bonus:.4f}, missed_opportunity_penalty={missed_opportunity_penalty:.4f}")
            logger.info(f"  direction_bonus={direction_bonus:.4f}, speed_bonus={speed_bonus:.4f}")
            logger.info(f"  agent_conf_penalty={agent_conf_penalty:.4f}, agent_classifier_agreement={agent_classifier_agreement:.4f}")
            logger.info(f"  total_reward={total_reward:.4f}")

        # Final safety clamps
        if pct_pnl <= -20.0:
            logger.warning(f"❌ Forcing reward cap for severe loss: {pct_pnl:.2f}")
            total_reward = min(total_reward, -2.0)
        elif -20.0 < pct_pnl < -2.0 and total_reward > 0:
            logger.warning(f"⚠️ Moderate loss with positive reward → Clamping to 0")
            total_reward = min(total_reward, 0.0)
        if pct_pnl > 1.5 and total_reward < 0:
            logger.warning(f"⚠️ Positive trade with negative reward → Clamping to 0")
            total_reward = max(total_reward, 0.0)

        return np.clip(total_reward, -10.0, 10.0)