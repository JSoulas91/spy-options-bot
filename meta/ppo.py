import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, List

from meta.meta_agent_info import get_meta_agent_dims
from config import META_MODEL_PATH
from utils.logger import bot_logger as logger


class DualHeadLSTM(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64, lstm_hid: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.lstm = nn.LSTM(state_dim, lstm_hid, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.shared = nn.Sequential(
            nn.Linear(lstm_hid, hidden),
            nn.ReLU()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x.unsqueeze(1))
        h = self.dropout(out[:, -1, :])
        h = self.shared(h)
        dir_logits = self.dir_head(h)
        conf = self.conf_head(h).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        return dir_logits, conf, value


class PPOAgent:
    def __init__(self,
                 state_dim: Optional[int] = None,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 conf_loss_w: float = 0.20,
                 entropy_coef_start: float = 1e-2,
                 target_entropy: Optional[float] = None,
                 grad_clip_norm: float = 1.0):
        loaded = False
        if state_dim is None:
            # ✅ FIXED: dynamically fetch correct state dim (was hardcoded 73)
            state_dim, _ = get_meta_agent_dims()

        self.net = DualHeadLSTM(state_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.entropy_coef = torch.tensor(entropy_coef_start)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.conf_loss_w = conf_loss_w
        self.grad_clip_norm = grad_clip_norm
        self.tgt_entropy = target_entropy if target_entropy is not None else -1.0

        if os.path.exists(META_MODEL_PATH):
            try:
                ckpt = torch.load(META_MODEL_PATH, map_location="cpu")
                self.net.load_state_dict(ckpt["net"])
                self.optimizer.load_state_dict(ckpt["opt"])
                ent_coef = ckpt.get("ent_coef", entropy_coef_start)
                self.entropy_coef = torch.tensor(ent_coef)

                model_input_dim = self.net.state_dim
                if model_input_dim != state_dim:
                    logger.warning("⚠️ PPO model input dim mismatch: saved %d vs current %d → reinitializing fresh model.",
                                   model_input_dim, state_dim)
                    self.net = DualHeadLSTM(state_dim)
                    self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
                else:
                    loaded = True
                    logger.info("📥 PPO model loaded.")
            except Exception as e:
                logger.warning("⚠️ Failed to load PPO model due to error: %s — reinitializing.", str(e))

        if not loaded:
            logger.warning("⚠️ No saved PPO model found – starting fresh.")

    def save(self, path: str = META_MODEL_PATH):
        device = next(self.net.parameters()).device
        state = {
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "ent_coef": self.entropy_coef.item()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        logger.info("💾 PPO model saved → %s", path)

    @torch.no_grad()
    def act(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dir_logits, conf, _ = self.net(state)
        dist = Categorical(logits=dir_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), conf.item(), log_prob, entropy

    @staticmethod
    def _discounted_returns(rewards: List[float], dones: List[int], last_value: float, gamma: float):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def train_step(
        self,
        states: torch.Tensor,
        actions_dir: torch.Tensor,
        target_conf: torch.Tensor,
        rewards: List[float],
        dones: List[int],
        next_states: torch.Tensor,
        old_logp: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = next(self.net.parameters()).device

        states = states.to(device)
        next_states = next_states.to(device)
        actions_dir = actions_dir.to(device)
        target_conf = target_conf.to(device)

        if weights is None:
            weights = torch.ones_like(actions_dir, dtype=torch.float32)
        weights = weights.to(device)

        with torch.no_grad():
            _, _, next_v = self.net(next_states)
            last_value = next_v.mean().item()

        _, _, values = self.net(states)
        returns = self._discounted_returns(rewards, dones, last_value, self.gamma).to(device)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dir_logits, conf_pred, value_pred = self.net(states)
        dist = Categorical(logits=dir_logits)
        log_probs = dist.log_prob(actions_dir)

        if old_logp is None:
            old_logp = log_probs.detach()

        ratios = torch.exp(log_probs - old_logp)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2)

        conf_loss = nn.functional.mse_loss(conf_pred, target_conf, reduction="none")
        value_loss = nn.functional.mse_loss(value_pred, returns, reduction="none")
        entropy = dist.entropy()
        entropy_loss = -self.entropy_coef * entropy

        total_loss = (
            policy_loss +
            self.conf_loss_w * conf_loss +
            value_loss +
            entropy_loss
        )

        total_loss = (total_loss * weights).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = (old_logp - log_probs).mean().item()

        logger.debug("Loss components — policy: %.6f  conf: %.6f  value: %.6f  entropy: %.6f  KL: %.6f",
                     policy_loss.mean().item(), conf_loss.mean().item(), value_loss.mean().item(),
                     entropy.mean().item(), approx_kl)

        return (advantages.detach() ** 2)