# meta/meta_env.py

import numpy as np
import random

class MetaEnv:
    def __init__(self, state_history):
        self.state_history = state_history
        self.reset()

    def reset(self):
        self.current_step = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.state_history):
            self.done = True
            return None
        state = self.state_history[self.current_step]
        return np.array([
            state.get("win_streak", 0),
            state.get("loss_streak", 0),
            state.get("vix", 20.0),
            state.get("volatility", 0.5),
            state.get("time_of_day", 10.0),
            state.get("trade_type", 0)
        ], dtype=np.float32)

    def step(self, action):
        # Define reward rules
        state = self.state_history[self.current_step]
        base_reward = state.get("reward", 0.0)

        # Example: penalize if wrong strategy mod or wrong size
        if action == 0:  # e.g. no change
            reward = base_reward
        elif action == 1:  # more aggressive
            reward = base_reward * 1.2 if state.get("outcome") == "win" else base_reward * 0.8
        elif action == 2:  # more conservative
            reward = base_reward * 0.9 if state.get("outcome") == "win" else base_reward * 1.1
        else:
            reward = base_reward  # fallback

        self.current_step += 1
        next_state = self._get_state()
        return next_state, reward, self.done, {}

    def sample_action(self):
        return random.choice([0, 1, 2])  # placeholder actions

    def action_space(self):
        return 3  # number of discrete actions

    def state_space(self):
        return 6  # number of elements in the state vector