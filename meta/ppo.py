# meta/ppo.py  – dual‑head (direction + confidence) PPO
import os, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
from meta.meta_agent_info import get_meta_agent_dims
from config  import META_MODEL_PATH
from utils.logger import bot_logger as logger

# ─────────────────────────────────────────────────────────
class DualHeadAC(nn.Module):
    """
    shared → dir_logits (3‑way)  &  conf_score (scalar 0‑1 via Sigmoid)
    """
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.dir_head   = nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, 3))          # logits
        self.conf_head  = nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, 1),
                                        nn.Sigmoid())                  # 0‑1
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.shared(x)
        return self.dir_head(h), self.conf_head(h).squeeze(-1), self.value_head(h)


# ─────────────────────────────────────────────────────────
class PPOAgent:
    """
    Handles **direction** (categorical) + **confidence** (regression) heads.
    """
    def __init__(
        self,
        state_dim: int | None = None,
        lr: float = 3e‑4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        conf_loss_w: float = 0.2,        # weight of confidence MSE
        entropy_coef_start: float = 1e‑2,
        target_entropy: float | None = None,
    ):
        if state_dim is None:
            state_dim, _ = get_meta_agent_dims()       # action dim no longer used
        self.net = DualHeadAC(state_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # entropy coefficient (for direction head only)
        self.entropy_coef = torch.tensor(entropy_coef_start, requires_grad=True)
        self.entropy_opt  = optim.Adam([self.entropy_coef], lr=1e‑3)

        self.gamma, self.eps_clip, self.k_epochs = gamma, eps_clip, k_epochs
        self.conf_loss_w = conf_loss_w
        self.tgt_entropy = target_entropy or -1.0 * 3   # 3 classes
        self._load()

    # ───── persistence ───────────────────────────────────
    def _load(self):
        if not os.path.exists(META_MODEL_PATH): return
        ckpt = torch.load(META_MODEL_PATH)
        self.net.load_state_dict(ckpt['net'])
        self.opt.load_state_dict(ckpt['opt'])
        self.entropy_coef.data = torch.tensor(ckpt.get('ent_coef', 1e‑2))
        self.entropy_opt.load_state_dict(ckpt['ent_opt'])
        logger.info("📥 Dual‑head PPO model loaded.")

    def save(self):
        torch.save({
            'net':       self.net.state_dict(),
            'opt':       self.opt.state_dict(),
            'ent_coef':  self.entropy_coef.item(),
            'ent_opt':   self.entropy_opt.state_dict(),
        }, META_MODEL_PATH)
        logger.info("💾 Dual‑head PPO saved.")

    # ───── rollout helpers ───────────────────────────────
    @torch.no_grad()
    def act(self, state: torch.Tensor):
        dir_logits, conf, _ = self.net(state)
        dist   = Categorical(logits=dir_logits)
        action = dist.sample()
        return (
            action.item(),                # 0/1/2
            conf.item(),                  # scalar 0‑1
            dist.log_prob(action),        # for PPO
            dist.entropy()
        )

    # ───── training utils ────────────────────────────────
    @staticmethod
    def _returns(rewards, dones, last_v, γ):
        ret, out = last_v, []
        for r, d in zip(reversed(rewards), reversed(dones)):
            ret = r + γ * ret * (1 - int(d))
            out.insert(0, ret)
        return torch.tensor(out, dtype=torch.float32)

    def _entropy_update(self, entropy):
        loss = -(self.entropy_coef * (entropy - self.tgt_entropy).detach())
        self.entropy_opt.zero_grad(); loss.backward(); self.entropy_opt.step()

    # ───── public train ‑ step (used by train_meta_agent.py) ──
    def train_step(self, batch):
        """
        batch = dict with
         states, next_states (Tensor[N,dim])
         actions_dir (LongTensor[N])
         actions_conf (FloatTensor[N]) ← target
         rewards, dones (list)
         old_logp   (Tensor[N])
         weights    (Tensor[N])        ← PER; can be 1’s
        """
        s, ns = batch['states'], batch['next_states']
        with torch.no_grad():
            _, _, v_next = self.net(ns)
        _, _, v      = self.net(s)

        returns = self._returns(batch['rewards'], batch['dones'], v_next, self.gamma)
        adv     = returns - v.detach()

        # forward current policy
        dir_logits, conf_pred, _ = self.net(s)
        dist   = Categorical(logits=dir_logits)
        logp   = dist.log_prob(batch['actions_dir'])
        ratio  = torch.exp(logp - batch['old_logp'])

        surr1  = ratio * adv
        surr2  = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
        actor_loss   = -torch.min(surr1, surr2).mean()
        critic_loss  = nn.MSELoss()(v.squeeze(), returns)
        conf_loss    = nn.MSELoss()(conf_pred, batch['actions_conf']) * self.conf_loss_w
        entropy_loss = -self.entropy_coef * dist.entropy().mean()

        loss = actor_loss + 0.5*critic_loss + conf_loss + entropy_loss
        if 'weights' in batch:
            loss = loss * batch['weights'].mean()

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self._entropy_update(dist.entropy().mean())

        # TD‑errors for PER
        return (returns - v.detach()).cpu().numpy().tolist()