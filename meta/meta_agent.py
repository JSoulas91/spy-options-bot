# meta/meta_agent.py

import json
import os
from datetime import datetime
from utils.logger import bot_logger as logger
from meta.ppo import PPOPolicy

META_STATE_PATH = os.path.join(os.path.dirname(__file__), 'meta_state.json')

class MetaAgent:
    def __init__(self):
        self.state = self.load_state()
        self.ppo = PPOPolicy(input_size=5, output_size=3)  # You can adjust later

    def load_state(self):
        if os.path.exists(META_STATE_PATH):
            with open(META_STATE_PATH, 'r') as f:
                return json.load(f)
        else:
            logger.warning("[MetaAgent] No meta_state.json found. Initializing new state.")
            return {
                "win_streak": 0,
                "loss_streak": 0,
                "last_action": None,
                "history": []
            }

    def save_state(self):
        with open(META_STATE_PATH, 'w') as f:
            json.dump(self.state, f, indent=4)

    def update_result(self, reward):
        """
        Call this after each trade outcome.
        reward: +1 for win, -1 for loss, 0 for neutral
        """
        if reward > 0:
            self.state["win_streak"] += 1
            self.state["loss_streak"] = 0
        elif reward < 0:
            self.state["loss_streak"] += 1
            self.state["win_streak"] = 0

        self.state["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "reward": reward,
            "win_streak": self.state["win_streak"],
            "loss_streak": self.state["loss_streak"]
        })

        if len(self.state["history"]) > 1000:
            self.state["history"] = self.state["history"][-1000:]

        self.save_state()

    def decide(self, vix, time_of_day, trade_type, volatility):
        """
        Converts input into a state vector and gets action from PPO.
        Action space (example):
            0 = No change
            1 = Raise confidence threshold
            2 = Reduce position size
        """
        input_state = [
            self.state["win_streak"] - self.state["loss_streak"],  # Net streak
            vix,
            time_of_day,    # e.g. 0.0–1.0 (normalized)
            trade_type,     # 0 = day trade, 1 = swing
            volatility      # e.g. ATR or intraday stddev
        ]

        action = self.ppo.select_action(input_state)
        self.state["last_action"] = action
        self.save_state()
        return action

# Example usage:
# meta_agent = MetaAgent()
# action = meta_agent.decide(vix=18, time_of_day=0.75, trade_type=0, volatility=0.015)
# meta_agent.update_result(reward=1)