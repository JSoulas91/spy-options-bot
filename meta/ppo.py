# meta/ppo.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from meta.meta_agent_info import get_meta_agent_dims
from config import META_MODEL_PATH
from utils.logger import bot_logger as logger

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

class PPOAgent:
    def __init__(self, state_dim=None, action_dim=None, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        if state_dim is None or action_dim is None:
            state_dim, action_dim = get_meta_agent_dims()
            logger.info(f"📦 Loaded PPO state/action dims from file: {state_dim}, {action_dim}")

        self.model = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.load()

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, META_MODEL_PATH)
        logger.info(f"💾 PPO model + optimizer saved to {META_MODEL_PATH}")

    def load(self):
        if os.path.exists(META_MODEL_PATH):
            checkpoint = torch.load(META_MODEL_PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"📥 PPO model + optimizer loaded from {META_MODEL_PATH}")
        else:
            logger.warning(f"⚠️ No saved PPO model found at {META_MODEL_PATH}. Starting fresh.")

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - int(done))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, memory):
        if not memory or len(memory['states']) == 0:
            logger.warning("⚠️ PPO update skipped: empty memory buffer.")
            return

        states = torch.stack(memory['states'])
        actions = torch.tensor(memory['actions'])
        old_log_probs = torch.stack(memory['log_probs'])
        returns = self.compute_returns(memory['rewards'], memory['dones'], memory['values'], memory['next_value'])

        for _ in range(self.K_epochs):
            log_probs = []
            state_values = []
            entropy_terms = []

            for state in states:
                probs, value = self.model(state.unsqueeze(0))
                dist = Categorical(probs)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()

                log_probs.append(log_prob)
                state_values.append(value)
                entropy_terms.append(entropy)

            log_probs = torch.stack(log_probs)
            values = torch.cat(state_values).squeeze()
            entropy = torch.stack(entropy_terms).mean()

            advantages = returns - values.detach()
            ratios = torch.exp(log_probs - old_log_probs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info("✅ PPO update complete.")