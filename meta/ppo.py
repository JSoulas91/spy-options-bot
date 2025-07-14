import psutil
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, List

from meta.meta_agent_info import get_meta_agent_dims
from config import META_MODEL_PATH
from utils.logger import bot_logger as logger
from utils.telegram_utils import send_telegram_message


def report_memory_usage(tag=""):
    """Logs and sends current RAM usage to Telegram."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    message = f"[{tag}] RAM usage: {mem_mb:.2f} MB"
    print(message)
    send_telegram_message(message)


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
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        h = self.shared(h)
        dir_logits = self.dir_head(h)
        conf = self.conf_head(h)
        value = self.value_head(h)
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
                 grad_clip_norm: float = 1.0,
                 use_kl_conf_penalty: bool = True,
                 kl_conf_coef: float = 0.01,
                 overconfidence_penalty: float = 0.05):
        loaded = False
        if state_dim is None:
            state_dim, _ = get_meta_agent_dims()

        self.net = DualHeadLSTM(state_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.entropy_coef = nn.Parameter(torch.tensor(entropy_coef_start))
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.conf_loss_w = conf_loss_w
        self.grad_clip_norm = grad_clip_norm
        self.tgt_entropy = target_entropy if target_entropy is not None else -1.0

        self.use_kl_conf_penalty = use_kl_conf_penalty
        self.kl_conf_coef = kl_conf_coef
        self.overconfidence_penalty = overconfidence_penalty

        if os.path.exists(META_MODEL_PATH):
            try:
                ckpt = torch.load(META_MODEL_PATH, map_location="cpu")
                self.net.load_state_dict(ckpt["net"])
                self.optimizer.load_state_dict(ckpt["opt"])
                ent_coef = ckpt.get("ent_coef", entropy_coef_start)
                self.entropy_coef = nn.Parameter(torch.tensor(ent_coef))

                if self.net.state_dim != state_dim:
                    logger.warning("⚠️ PPO model input dim mismatch — reinitializing.")
                    self.net = DualHeadLSTM(state_dim)
                    self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
                else:
                    loaded = True
                    logger.info("📥 PPO model loaded.")
            except Exception as e:
                logger.warning("⚠️ Failed to load PPO model: %s — reinitializing.", str(e))

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
        weights: Optional[torch.Tensor] = None,
        prev_conf: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        conf_mask: Optional[torch.Tensor] = None,
        confidence_penalty_mask: Optional[torch.Tensor] = None,
        mini_batch_size: int = 32
    ) -> torch.Tensor:
        device = next(self.net.parameters()).device
        states, next_states = states.to(device), next_states.to(device)
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
    
        total_size = states.size(0)
        indices = torch.randperm(total_size)
    
        for i in range(0, total_size, mini_batch_size):
            batch_idx = indices[i:i + mini_batch_size]
        
            s = states[batch_idx]
            a = actions_dir[batch_idx]
            c = target_conf[batch_idx]
        
            r = returns[batch_idx]
            if r.ndim > 1:
                r = r.squeeze(-1)
        
            adv = advantages[batch_idx]
            if adv.ndim > 1:
                adv = adv.squeeze(-1)
        
            w = weights[batch_idx]
            if w.ndim > 1:
                w = w.squeeze(-1)
        
            olp = old_logp[batch_idx] if old_logp is not None else None
            if olp is not None and olp.ndim > 1:
                olp = olp.squeeze(-1)
        
            pcm = confidence_penalty_mask[batch_idx] if confidence_penalty_mask is not None else None
            cm = conf_mask[batch_idx] if conf_mask is not None else None
            pconf = prev_conf[batch_idx] if prev_conf is not None else None
    
            dir_logits, conf_pred, value_pred = self.net(s)
    
            dist = Categorical(logits=dir_logits)
            log_probs = dist.log_prob(a)
    
            if olp is None:
                olp = log_probs.detach()
    
            ratios = torch.exp(log_probs - olp)
    
            # ✅ Ensure shape match between ratios and adv
            if adv.ndim == 2:
                adv = adv.squeeze(-1)
            elif adv.ndim == 0:
                adv = adv.unsqueeze(0)
    
            # ✅ PPO surrogate objective
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2)
            policy_loss = (policy_loss * w).mean()
    
            # Confidence loss
            conf_pred = conf_pred.squeeze(-1)
            conf_loss = nn.functional.mse_loss(conf_pred, c, reduction="none")
    
            if cm is not None:
                cm = cm.to(conf_pred.device)
                conf_loss = (conf_loss * cm).sum() / (cm.sum() + 1e-8)
            else:
                conf_loss = conf_loss.mean()
    
            if pcm is not None:
                pcm = pcm.to(conf_pred.device)
                penalty_loss = (conf_pred ** 2) * pcm
                penalty_loss = penalty_loss.sum() / (pcm.sum() + 1e-8)
                conf_loss += self.overconfidence_penalty * penalty_loss
    
            # Value loss
            value_loss = nn.functional.smooth_l1_loss(value_pred.squeeze(-1), r, reduction="none")
            value_loss = (value_loss * w).mean()
    
            # Entropy bonus
            entropy = dist.entropy()
            entropy_loss = -self.entropy_coef * (entropy * w).mean()
    
            total_loss = (
                policy_loss +
                self.conf_loss_w * conf_loss +
                value_loss +
                entropy_loss
            )
    
            if self.use_kl_conf_penalty and pconf is not None:
                pconf = pconf.to(device)
                kl_conf = nn.functional.mse_loss(conf_pred.detach(), pconf, reduction="none").squeeze(-1)
                kl_penalty = self.kl_conf_coef * (kl_conf * w).mean()
                total_loss += kl_penalty
    
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
            self.optimizer.step()
    
        return advantages.detach() ** 2