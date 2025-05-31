import os
import numpy as np
from meta.ppo import PPOPolicy
from meta.meta_state import get_current_meta_state

class MetaAgent:
    def __init__(self, model_path="meta/meta_agent_model.pkl"):
        self.model_path = model_path
        self.policy = self._load_or_initialize_policy()

    def _load_or_initialize_policy(self):
        policy = PPOPolicy(input_dim=5, output_dim=3)
        if os.path.exists(self.model_path):
            try:
                policy.load(self.model_path)
                print("[MetaAgent] Loaded existing meta-agent policy.")
            except Exception as e:
                print(f"[MetaAgent] Failed to load existing policy: {e}. Starting fresh.")
        else:
            print("[MetaAgent] No existing policy found. Initializing new one.")
        return policy

    def select_action(self, state):
        """
        Given a meta-state, return an action (as index).
        """
        state = np.array(state).reshape(1, -1)
        return self.policy.select_action(state)

    def interpret_action(self, action_index):
        """
        Converts action index into a dictionary of strategy parameters.
        """
        actions = {
            0: {"confidence_threshold": 0.65, "position_size": 0.05},  # Conservative
            1: {"confidence_threshold": 0.55, "position_size": 0.10},  # Moderate
            2: {"confidence_threshold": 0.45, "position_size": 0.15},  # Aggressive
        }
        return actions.get(action_index, actions[1])  # Default to moderate

    def save_policy(self):
        """
        Save the trained model.
        """
        self.policy.save(self.model_path)
        print("[MetaAgent] Meta-agent policy saved.")