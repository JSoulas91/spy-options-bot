# meta/ppo.py  – Dual‑head PPO with tiny‑LSTM encoder + Dropout + Grad‑clip
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from meta.meta_agent_info import get_meta_agent_dims
from config                import META_MODEL_PATH
from utils.logger          import bot_logger as logger

# ──────────────────────────────────────────────────────────────
class DualHeadLSTM(nn.Module):
    """
    state(feat) → LSTM → Dropout → shared FC
                     ├── dir_head  (3 logits)
                     ├── conf_head (sigmoid 0‑1)
                     └── value_head
    """
    def __init__(self, state_dim: int, hidden: int = 64, lstm_hid: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hid,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.shared  = nn.Sequential(
            nn.Linear(lstm_hid, hidden),
            nn.ReLU()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)          # logits
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
            nn.Linear(hidden, 1)          # scalar value
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, feat) → (batch, 1, feat)
        out, _ = self.lstm(x.unsqueeze(1))
        h = self.dropout(out[:, -1, :])
        h = self.shared(h)
        return (
            self.dir_head(h),              # logits
            self.conf_head(h).squeeze(-1), # scalar conf
            self.value_head(h)             # critic value
        )

# ──────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Dual‑output PPO agent:
      • Direction (categorical 0,1,2)
      • Confidence (regression 0–1)
    """
    def __init__(
        self,
        state_dim: int | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        conf_loss_w: float = 0.20,
        entropy_coef_start: float = 1e-2,
        target_entropy: float | None = None,
        grad_clip_norm: float = 1.0,
    ):
        if state_dim is None:
            state_dim, _ = get_meta_agent_dims()
        self.net = DualHeadLSTM(state_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.entropy_coef = torch.tensor(entropy_coef_start, requires_grad=True)
        self.entropy_opt  = optim.Adam([self.entropy_coef], lr=1e-3)

        self.gamma, self.eps_clip = gamma, eps_clip
        self.k_epochs     = k_epochs
        self.conf_loss_w  = conf_loss_w
        self.grad_clip_norm = grad_clip_norm
        self.tgt_entropy  = target_entropy or -1.0 * 3  # action space=3
        self._load()

    # ───────── persistence ────────────────────────────────
    def _load(self):
        if not os.path.exists(META_MODEL_PATH):
            logger.warning("⚠️ No saved PPO model found – starting fresh.")
            return
        ckpt = torch.load(META_MODEL_PATH, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.entropy_coef.data = torch.tensor(ckpt.get("ent_coef", 1e-2))
        self.entropy_opt.load_state_dict(ckpt["ent_opt"])
        logger.info("📥 PPO model loaded from disk.")

    def save(self):
        torch.save(
            {
                "net":       self.net.state_dict(),
                "opt":       self.opt.state_dict(),
                "ent_coef":  self.entropy_coef.item(),
                "ent_opt":   self.entropy_opt.state_dict(),
            },
            META_MODEL_PATH,
        )
        logger.info("💾 PPO model saved → %s", META_MODEL_PATH)

    # ───────── rollout (inference) ─────────────────────────
    @torch.no_grad()
    def act(self, state: torch.Tensor):
        dir_logits, conf, _ = self.net(state)
        dist   = Categorical(logits=dir_logits)
        action = dist.sample()
        return (
            action.item(),           # 0/1/2
            conf.item(),             # 0‑1
            dist.log_prob(action),   # Tensor
            dist.entropy(),          # Tensor
        )

    # ───────── PPO internals ──────────────────────────────
    @staticmethod
    def _returns(rewards, dones, last_v, gamma):
        """
        Vectorised discounted return per batch element.
        rewards, dones: list[float/int] length T
        last_v: tensor shape [B] (B=batch size)
        Returns: tensor shape [T, B]
        """
        # Convert scalars to tensors on CPU
        R = last_v.clone()  # shape [B]
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            r = torch.tensor(r, dtype=torch.float32)
            d = torch.tensor(d, dtype=torch.float32)
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.stack(returns)        # shape [T, B]

    def _entropy_update(self, entropy_mean: torch.Tensor):
        loss = -(self.entropy_coef * (entropy_mean - self.tgt_entropy).detach())
        self.entropy_opt.zero_grad()
        loss.backward()
        self.entropy_opt.step()

    # ───────────────────────────────────────────────────────
    def train_step(
        self,
        states: torch.Tensor,
        actions_dir: torch.Tensor,     # LongTensor[N]
        target_conf: torch.Tensor,     # FloatTensor[N]
        rewards: list,
        dones: list,
        next_states: torch.Tensor,
        old_logp: torch.Tensor,
        weights: torch.Tensor | None = None,
    ):
        """
        Single gradient update on CPU tensors.
        """
        with torch.no_grad():
            _, _, v_next = self.net(next_states)          # shape [B,1]
        _, _, v = self.net(states)                        # shape [B,1]

        # returns shape [T,B], squeeze to [B] because T==1 for 1‑step returns
        returns = self._returns(rewards, dones, v_next.squeeze(), self.gamma)
        returns = returns.squeeze()                       # shape [B]

        adv = returns - v.squeeze().detach()

        dir_logits, conf_pred, _ = self.net(states)
        dist   = Categorical(logits=dir_logits)
        logp   = dist.log_prob(actions_dir)
        ratio  = torch.exp(logp - old_logp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(v.squeeze(), returns)
        conf_loss   = nn.MSELoss()(conf_pred, target_conf) * self.conf_loss_w
        entropy_term= dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss + conf_loss - self.entropy_coef * entropy_term
        if weights is not None:
            loss *= weights.mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
        self.opt.step()

        self._entropy_update(entropy_term)

        td_error = (returns - v.detach().squeeze()).cpu().numpy().tolist()
        return td_error