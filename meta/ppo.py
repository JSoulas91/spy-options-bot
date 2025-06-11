# meta/ppo.py – Dual‑head PPO (batch‑safe returns + stable TD‑error)
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
                      ├── dir_head  (3‑logits)
                      ├── conf_head (sigmoid 0‑1)
                      └── value_head (scalar V)
    """
    def __init__(self, state_dim: int, hidden: int = 64, lstm_hid: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, lstm_hid, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.shared  = nn.Sequential(nn.Linear(lstm_hid, hidden), nn.ReLU())

        self.dir_head  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 3))           # logits
        self.conf_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1), nn.Sigmoid())
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 1))          # scalar

    def forward(self, x: torch.Tensor):
        # x: [B, feat]  → [B, 1, feat]
        out, _ = self.lstm(x.unsqueeze(1))
        h = self.dropout(out[:, -1, :])
        h = self.shared(h)
        return (
            self.dir_head(h),               # [B,3] logits
            self.conf_head(h).squeeze(-1),  # [B]
            self.value_head(h)              # [B,1]
        )

# ──────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Dual‑output PPO agent:
      • Direction (categorical 0/1/2)
      • Confidence (regression 0‑1)
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
        self.k_epochs       = k_epochs
        self.conf_loss_w    = conf_loss_w
        self.grad_clip_norm = grad_clip_norm
        self.tgt_entropy    = target_entropy or -3.0  # 3‑class action space
        self._load()

    # ───────── persistence ─────────────────────────────────
    def _load(self):
        if not os.path.exists(META_MODEL_PATH):
            logger.warning("⚠️ No saved PPO model found – starting fresh.")
            return
        ckpt = torch.load(META_MODEL_PATH, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.entropy_coef.data.copy_(torch.tensor(ckpt.get("ent_coef", 1e-2)))
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
        logits, conf, _ = self.net(state)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return (
            action.item(),            # 0/1/2
            conf.item(),              # confidence 0‑1
            dist.log_prob(action),    # tensor
            dist.entropy(),           # tensor
        )

    # ───────── helper: one‑step discounted returns ─────────
    @staticmethod
    def _discounted_returns(rewards, dones, last_v, gamma):
        """
        rewards, dones: list[float] length B
        last_v        : Tensor[B]
        returns       : Tensor[B]
        """
        r = torch.tensor(rewards, dtype=torch.float32)
        d = torch.tensor(dones,   dtype=torch.float32)
        return r + gamma * last_v * (1 - d)

    # ───────── entropy‑temperature update ──────────────────
    def _entropy_update(self, entropy_mean):
        loss = -(self.entropy_coef * (entropy_mean - self.tgt_entropy).detach())
        self.entropy_opt.zero_grad()
        loss.backward()
        self.entropy_opt.step()

    # ───────── single training step ────────────────────────
    def train_step(
        self,
        states: torch.Tensor,          # [B,feat]
        actions_dir: torch.Tensor,     # [B] long
        target_conf: torch.Tensor,     # [B] float
        rewards: list,                 # len B
        dones: list,                   # len B
        next_states: torch.Tensor,     # [B,feat]
        old_logp: torch.Tensor,        # [B] float
        weights: torch.Tensor | None = None,  # [B]
    ):
        # 1. Forward passes
        with torch.no_grad():
            _, _, v_next = self.net(next_states)           # [B,1]
        _, _, v = self.net(states)                         # [B,1]

        # 2. Returns & advantage
        returns = self._discounted_returns(
            rewards, dones, v_next.squeeze(-1), self.gamma
        )
        advantages = returns - v.squeeze(-1).detach()

        # 3. Log‑probs with current policy
        logits_dir, conf_pred, _ = self.net(states)
        dist_dir = Categorical(logits=logits_dir)
        logp_dir = dist_dir.log_prob(actions_dir)

        # Combined log‑prob (direction + regression head)
        # For simplicity we treat conf head with MSE not log‑prob
        ratio = torch.exp(logp_dir