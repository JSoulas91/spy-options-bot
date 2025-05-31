# meta/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
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
        probs, _ = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.model = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - int(done))
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, memory):
        states = torch.stack(memory['states'])
        actions = torch.tensor(memory['actions'])
        old_log_probs = torch.stack(memory['log_probs'])
        returns = self.compute_returns(memory['rewards'], memory['dones'], memory['values'], memory['next_value'])
        returns = returns.detach()
        
        for _ in range(self.K_epochs):
            log_probs_list = []
            state_values_list = []
            dist_entropy_list = []

            for state in states:
                probs, value = self.model(state)
                dist = Categorical(probs)
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()

                log_probs_list.append(log_prob)
                state_values_list.append(value)
                dist_entropy_list.append(entropy)

            log_probs = torch.stack(log_probs_list)
            values = torch.cat(state_values_list).squeeze()
            entropy = torch.stack(dist_entropy_list).mean()

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