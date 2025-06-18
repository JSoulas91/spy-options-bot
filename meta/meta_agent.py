import os
import numpy as np
import torch

from meta.ppo import PPOAgent
from utils.logger import bot_logger as logger
from config import META_MODEL_PATH

class MetaAgent:
    def __init__(self, ckpt_path: str = META_MODEL_PATH):
        self.ckpt_path = ckpt_path
        self._model = PPOAgent()  # auto-loads state/action dims internally
        if os.path.exists(ckpt_path):
            logger.info(f"[MetaAgent] Loaded model from {ckpt_path}")
        else:
            logger.info("[MetaAgent] No checkpoint found; initialized new PPOAgent.")
        self._model.net.eval()
        logger.info("[MetaAgent] PPO model ready.")

    def select_action(self, state: np.ndarray | list):
        """
        Returns:
            action_idx (int)  : 0 = veto / 1 = normal / 2 = tighten-exit
            agent_conf (float): model confidence 0-1 (sigmoid head)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act_idx, conf, _, _ = self._model.act(state_t)
        return int(act_idx), float(conf)

    @staticmethod
    def interpret_action(action_idx: int, agent_conf: float):
        mapping = {
            0: {"confidence_threshold": 0.80, "scale_factor": 0.5, "tighten_exit": True},
            1: {"confidence_threshold": 0.60, "scale_factor": 1.0, "tighten_exit": False},
            2: {"confidence_threshold": 0.50, "scale_factor": 0.8, "tighten_exit": True},
        }
        out = mapping.get(action_idx, mapping[1])
        out["agent_confidence"] = agent_conf
        return out

    def veto_retry(self, state):
        return self.select_action(state)[0] == 0

    def save(self):
        self._model.save(self.ckpt_path)
        logger.info(f"[MetaAgent] Model saved to {self.ckpt_path}")

    def save_model(self):  # ← Added for compatibility with training code
        self.save()

# singleton helper
_meta_agent_instance = None
def get_meta_agent():
    global _meta_agent_instance
    if _meta_agent_instance is None:
        _meta_agent_instance = MetaAgent()
    return _meta_agent_instance

def should_retry_trade(contract: dict) -> bool:
    meta_agent = get_meta_agent()
    state = contract.get("meta_state")
    return False if state is None else not meta_agent.veto_retry(state)