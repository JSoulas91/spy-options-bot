import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from meta.meta_agent_info import get_meta_agent_dims
from config import META_MODEL_PATH
from utils.logger import bot_logger as logger


class DualHeadLSTM(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64, lstm_hid: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, lstm_hid, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.shared = nn.Sequential(nn.Linear(lstm_hid, hidden), nn.ReLU())

        self.dir_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 3))  # logits for 3 actions
        self.conf_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1), nn.Sigmoid())
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x.unsqueeze(1))  # [B,1,lstm_hid]
        h = self.dropout(out[:, -1, :])     # last timestep
        h = self.shared(h)
        return self.dir_head(h), self.conf_head(h).squeeze(-1), self.value_head(h).squeeze(-1)


class PPOAgent:
    def __init__(self,
                 state_dim: int | None = None,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 conf_loss_w: float = 0.20,
                 entropy_coef_start: float = 1e-2,
                 target_entropy: float | None = None,
                 grad_clip_norm: float = 1.0):
        if state_dim is None:
            state_dim, _ = get_meta_agent_dims()

        self.net = DualHeadLSTM(state_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.entropy_coef = torch.tensor(entropy_coef_start, requires_grad=True)
        self.entropy_opt = optim.Adam([self.entropy_coef], lr=1e-3)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.conf_loss_w = conf_loss_w
        self.grad_clip_norm = grad_clip_norm
        self.tgt_entropy = target_entropy or -1.0  # Encourage higher exploration initially

        self.load()

    def load(self, path: str = META_MODEL_PATH):
        if not os.path.exists(path):
            logger.warning("⚠️ No saved PPO model found – starting fresh.")
            return
        ckpt = torch.load(path, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.entropy_coef.data.copy_(torch.tensor(ckpt.get("ent_coef", 1e-2)))
        self.entropy_opt.load_state_dict(ckpt["ent_opt"])
        logger.info("📥 PPO model loaded.")

    def save(self):
        os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "ent_coef": self.entropy_coef.item(),
            "ent_opt": self.entropy_opt.state_dict(),
        }, META_MODEL_PATH)
        logger.info("💾 PPO model saved → %s", META_MODEL_PATH)

    @torch.no_grad()
    def act(self, state: torch.Tensor):
        dir_logits, conf, _ = self.net(state)
        dist = Categorical(logits=dir_logits)
        action = dist.sample()
        return action.item(), conf.item(), dist.log_prob(action), dist.entropy()

    @staticmethod
    def _discounted_returns(rewards, dones, last_value, gamma):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def _entropy_update(self, entropy_mean):
        # Encourage entropy above target
        ent_loss = -self.entropy_coef * (entropy_mean - self.tgt_entropy)
        self.entropy_opt.zero_grad()
        ent_loss.backward()
        self.entropy_opt.step()

    def train_step(self,
                   states: torch.Tensor,
                   actions_dir: torch.Tensor,
                   target_conf: torch.Tensor,
                   rewards: list,
                   dones: list,
                   next_states: torch.Tensor,
                   old_logp: torch.Tensor,
                   weights: torch.Tensor | None = None):

        device = next(self.net.parameters()).device

        states = states.to(device)
        next_states = next_states.to(device)
        actions_dir = actions_dir.to(device)
        target_conf = target_conf.to(device)
        old_logp = old_logp.to(device)

        # Compute returns
        with torch.no_grad():
            _, _, next_v = self.net(next_states)
            last_value = next_v.mean().item()

        _, _, values = self.net(states)
        returns = self._discounted_returns(rewards, dones, last_value, self.gamma).to(device)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Forward pass
        dir_logits, conf_pred, value_pred = self.net(states)
        dist = Categorical(logits=dir_logits)
        logp = dist.log_prob(actions_dir)
        entropy = dist.entropy().mean()

        # PPO clipped objective
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.MSELoss()(value_pred, returns)
        conf_loss = nn.MSELoss()(conf_pred, target_conf)

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss + self.conf_loss_w * conf_loss - self.entropy_coef * entropy

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.opt.step()

        self._entropy_update(entropy.detach())

        # TD error for PER
        td_error = (value_pred.detach() - returns).abs()
        return td_error