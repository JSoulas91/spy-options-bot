# meta/meta_agent.py
"""
Wrapper around the dual‑head PPO model.
• select_action(state) ➜ (action_index, agent_confidence)
• interpret_action(idx, conf) ➜ strategy parameters dict
"""

import os
import numpy as np
import torch

from meta.ppo import PPOAgent          # ← our tiny‑LSTM PPO
from utils.logger import bot_logger as logger
from config import META_MODEL_PATH

class MetaAgent:
    def __init__(self, ckpt_path: str = META_MODEL_PATH):
        self.ckpt_path = ckpt_path
        self._model    = PPOAgent()          # dims auto‑loaded
        self._model.net.eval()               # inference mode
        logger.info("[MetaAgent] PPO model ready.")

    # ───────────────────────────────────────────────
    def select_action(self, state: np.ndarray | list):
        """
        Returns:
            action_idx (int)  : 0 = veto / 1 = normal / 2 = tighten‑exit
            agent_conf (float): model’s confidence 0‑1 (from sigmoid head)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act_idx, conf, _, _ = self._model.act(state_t)
        return int(act_idx), float(conf)

    # ───────────────────────────────────────────────
    @staticmethod
    def interpret_action(action_idx: int, agent_conf: float):
        """
        Map action index → dict of strategy knobs + echo confidence.
        """
        mapping = {
            0: {                         # force exit / veto
                "confidence_threshold": 0.80,
                "scale_factor": 0.5,
                "tighten_exit": True,
            },
            1: {                         # neutral / default
                "confidence_threshold": 0.60,
                "scale_factor": 1.0,
                "tighten_exit": False,
            },
            2: {                         # aggressive but with tighter risk
                "confidence_threshold": 0.50,
                "scale_factor": 0.8,     # partial scale‑in
                "tighten_exit": True,
            },
        }
        out = mapping.get(action_idx, mapping[1])
        out["agent_confidence"] = agent_conf
        return out

    # ───────────────────────────────────────────────
    # helpers used elsewhere (optional)
    def veto_retry(self, state):
        idx, _ = self.select_action(state)
        return idx == 0

    def save(self):
        self._model.save()