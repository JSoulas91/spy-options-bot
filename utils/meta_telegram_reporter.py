# utils/meta_telegram_reporter.py
"""
Send concise meta‑agent training summaries and reward charts to Telegram.
"""

import io, time, matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict

from utils.logger       import bot_logger as logger
from telegram_bot       import TelegramBot

TG = TelegramBot()
LAST_SENT = 0.0
MIN_INTERVAL_SEC = 60  # avoid spam

def _make_reward_plot(reward_history: List[float]) -> bytes:
    """Return PNG bytes of reward line chart."""
    fig, ax = plt.subplots()
    ax.plot(reward_history, linewidth=2)
    ax.set_title("Meta‑Agent Reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Reward")
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def send_training_report(stats: Dict, reward_history: List[float]):
    """
    stats = {
        "epoch": 30,
        "avg_reward": 0.48,
        "reward_std": 0.17,
        "sharpe": 2.8,
        "reject_rate": 0.42,
        "avg_duration": 8.3,
        "avg_pnl": 12.85,
    }
    reward_history = [ ... ]  # same len as epochs
    """
    global LAST_SENT
    if time.time() - LAST_SENT < MIN_INTERVAL_SEC:
        return  # throttle

    text = (
        f"📈 *Meta Training Update*\n"
        f"Epoch: *{stats['epoch']}*\n"
        f"Avg Reward: *{stats['avg_reward']:.3f}*\n"
        f"Sharpe (est): *{stats['sharpe']:.2f}*\n"
        f"Reject Rate: *{stats['reject_rate']*100:.1f}%*\n"
        f"Avg Trade Dur: *{stats['avg_duration']:.1f} min*\n"
        f"Avg PnL: *{stats['avg_pnl']:.2f}$*"
    )

    # send text first
    TG.send_message(text)

    # attach chart
    try:
        png_bytes = _make_reward_plot(reward_history)
        TG.send_photo(png_bytes, caption="Reward trend")
    except Exception as e:
        logger.error(f"[Meta Report] Plot send failed: {e}")

    LAST_SENT = time.time()