"""
Send concise meta‑agent training summaries (text + reward chart) to Telegram.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt

from utils.logger import bot_logger as logger
from utils.telegram_utils import send_telegram_message  # ✅ existing helper

_MIN_INTERVAL = 90  # seconds – throttle updates
_LAST_SENT_TS = 0.0


# ─────────────────────────────────────────────────────────────
def _make_chart(rewards: List[float]) -> bytes:
    """Return a PNG bytes object of the reward‑history chart."""
    fig, ax = plt.subplots()
    ax.plot(rewards, linewidth=2)
    ax.set_title("Meta‑Agent Avg Reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
def _fmt_stats(stats: Dict) -> str:
    """Format the stats dict into a Telegram‑friendly markdown message."""
    return (
        "🏋️‍♂️ *Meta‑Training Update*\n"
        f"• Epoch: *{stats.get('epoch', '?')}*\n"
        f"• Avg Reward: *{stats.get('avg_reward', 0):.3f}*\n"
        f"• Sharpe: *{stats.get('sharpe', 0):.2f}*\n"
        f"• Reject‑rate: *{stats.get('reject_rate', 0)*100:.1f}%*\n"
        f"• Avg Dur: *{stats.get('avg_duration', 0):.1f} min*\n"
        f"• Avg PnL: *{stats.get('avg_pnl', 0):.2f} $*\n"
        f"_UTC {datetime.utcnow().strftime('%Y‑%m‑%d %H:%M')}_"
    )


# ─────────────────────────────────────────────────────────────
def send_training_report(stats: Dict, rewards: List[float]) -> None:
    """
    Push a text summary (and optional reward chart) to Telegram.

    Parameters
    ----------
    stats   : dict   – keys like epoch, avg_reward, reward_std, sharpe, ...
    rewards : list   – running list of average reward per epoch
    """
    global _LAST_SENT_TS
    if time.time() - _LAST_SENT_TS < _MIN_INTERVAL:
        return  # anti‑spam throttle

    try:
        # 1) Send text message
        send