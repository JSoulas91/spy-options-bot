# meta/ppo.py  – dual‑head (direction + confidence) PPO  ✚  tiny‑LSTM encoder
import os, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

from meta.meta_agent_info import get_meta_agent_dims
from config                import META_MODEL_PATH
from utils.logger          import bot_logger as logger

# ──────────────────────────────────────────────────────────────
class DualHeadLSTM(nn.Module):
    """
    Architecture
    ┌───────────────────────────┐
    │  state  (batch, feat)     │
    └───────┬───────────────────┘
            │  unsqueeze(1)  →  sequence len = 1
            ▼
    ┌───────────────────────────┐
    │  LSTM (in=feat, hid=32)   │   «just 1 step, so super‑light»
    └───────┬───────────────────┘
            ▼   take last hidden (batch, 32)
    ┌───────────────────────────┐
    │  FC → ReLU  (shared)      │
    └───────┬───────────────────┘
       ┌────┴─────────┬───────────────┐
       ▼              ▼               ▼
  dir_head        conf_head       value_head
  (3‑logits)      (sigmoid)         (scalar)
    """
    def __init__(self, state_dim: int, hidden: int = 64, lstm_hid: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim,
                            hidden_size=lstm_hid,
                            num_layers=1,
                            batch_first=True)

        self.shared = nn.Sequential(
            nn.Linear(lstm_hid, hidden),
            nn.ReLU()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)            # 3 direction classes
        )
        self.conf_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()                    # 0‑1 confidence
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, feat) ➜ (batch, 1, feat)
        out, _ = self.lstm(x.unsqueeze(1))
        lstm_last = out[:, -1, :]          # (batch, lstm_hid)
        h = self.shared(lstm_last)
        return (
            self.dir_head(h),              # logits
            self.conf_head(h).squeeze(-1), # scalar
            self.value_head(h)             # critic value
        )

# ──────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Dual output heads          :   • direction  (categorical: 0/1/2)
                                   • confidence (continuous 0‑1)
    Tiny‑LSTM encoder in front :   better temporal context with negligible cost.
    """
    def __init__(
        self,
        state_dim: int | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        conf_loss_w: float = 0.2,        # weight of confidence MSE
        entropy_coef_start: float = 1e‑2,
        target_entropy: float | None = None,
    ):
        if state_dim is None:
            state_dim, _ = get_meta_agent_dims()   # action dim not needed now
        self.net = DualHeadLSTM(state_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # learnable entropy coefficient (direction head only)
        self.entropy_coef = torch.tensor(entropy_coef_start, requires_grad=True)
        self.entropy_opt  = optim.Adam([self.entropy_coef], lr=1e‑3)

        self.gamma, self.eps_clip, self.k_epochs = gamma, eps_clip, k_epochs
        self.conf_loss_w = conf_loss_w
        self.tgt_entropy = target_entropy or -1.0 * 3   # 3‑class target
        self._load()                                   # load weights if any

    # ───────── persistence ────────────────────────────────
    def _load(self):
        if not os.path.exists(META_MODEL_PATH):
            logger.warning("⚠️ No saved PPO model found – starting fresh.")
            return
        ckpt = torch.load(META_MODEL_PATH, map_location="cpu")
        self.net.load_state_dict(ckpt['net'])
        self.opt.load_state_dict(ckpt['opt'])
        self.entropy_coef.data = torch.tensor(ckpt.get('ent_coef', 1e‑2))
        self.entropy_opt.load_state_dict(ckpt['ent_opt'])
        logger.info("📥 PPO dual‑head + LSTM model loaded.")

    def save(self):
        torch.save({
            'net':       self.net.state_dict(),
            'opt':       self.opt.state_dict(),
            'ent_coef':  self.entropy_coef.item(),
            'ent_opt':   self.entropy_opt.state_dict(),
        }, META_MODEL_PATH)
        logger.info("💾 PPO model saved → %s", META_MODEL_PATH)

    # ───────── rollout helpers (used in live code) ─────────
    @torch.no_grad()
    def act(self, state: torch.Tensor):
        dir_logits, conf, _ = self.net(state)
        dist   = Categorical(logits=dir_logits)
        action = dist.sample()
        return (
            action.item(),                 # 0/1/2
            conf.item(),                   # scalar 0‑1
            dist.log_prob(action),         # log‑prob for PPO
            dist.entropy()
        )

    # ───────── PPO plumbing  ───────────────────────────────
    @staticmethod
    def _returns(rewards, dones, last_v, γ):
        ret, out = last_v, []
        for r, d in zip(reversed(rewards), reversed(dones)):
            ret = r + γ * ret * (1 - int(d))
            out.insert(0, ret)
        return torch.tensor(out, dtype=torch.float32)

    def _entropy_update(self, entropy_mean: torch.Tensor):
        loss = -(self.entropy_coef * (entropy_mean - self.tgt_entropy).detach())
        self.entropy_opt.zero_grad()
        loss.backward()
        self.entropy_opt.step()

    # public API called from train_meta_agent.py  ───────────
    def train_step(self, states, actions_dir, target_conf,
                   rewards, dones, next_states,
                   old_logp, weights=None):
        """
        All tensors are already on CPU and correct dtype.
        • states / next_states :  (N, feat)
        • actions_dir         :  LongTensor[N]
        • target_conf         :  FloatTensor[N]   (0‑1)
        • rewards / dones     :  list/array
        • old_logp            :  Tensor[N]
        • weights             :  PER importance (or None)
        """
        with torch.no_grad():
            _, _, v_next = self.net(next_states)
        _, _, v = self.net(states)                     # critic on current

        returns = self._returns(rewards, dones, v_next.squeeze(), self.gamma)
        adv     = returns - v.squeeze().detach()

        # forward current policy
        dir_logits, conf_pred, _ = self.net(states)
        dist  = Categorical(logits=dir_logits)
        logp  = dist.log_prob(actions_dir)
        ratio = torch.exp(logp - old_logp)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(v.squeeze(), returns)
        conf_loss   = nn.MSELoss()(conf_pred, target_conf) * self.conf_loss_w
        entropy_term= dist.entropy().mean()

        loss = actor_loss + 0.5*critic_loss + conf_loss - self.entropy_coef * entropy_term
        if weights is not None:
            loss = loss * weights.mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._entropy_update(entropy_term)

        # TD‑errors for PER priority update
        td_err = (returns - v.detach().squeeze()).cpu().numpy().tolist()
        return td_err