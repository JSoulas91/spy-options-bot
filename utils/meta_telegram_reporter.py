# utils/meta_telegram_reporter.py
"""
Send concise meta‑agent training summaries (text + reward chart) to Telegram.
"""
from __future__ import annotations
import io, time
from typing import List, Dict

import matplotlib.pyplot as plt

from utils.logger         import bot_logger as logger
from telegram_bot         import TelegramBot   # your existing wrapper

TG              = TelegramBot()
_MIN_INTERVAL   = 90          # seconds – throttle updates
_LAST_SENT_TS   = 0.0

# ─────────────────────────────────────────────────────────────
def _make_chart(rewards: List[float]) -> bytes:
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
def send_training_report(stats: Dict, rewards: List[float]):
    """
    stats keys:
        epoch, avg_reward, reward_std, sharpe, reject_rate,
        avg_duration, avg_pnl
    """
    global _LAST_SENT_TS
    if time.time() - _LAST_SENT_TS < _MIN_INTERVAL:
        return                                              # anti‑spam

    txt = (
        "📈 *Meta‑Training Update*\n"
        f"• Epoch: *{stats['epoch']}*\n"
        f"• Avg Reward: *{stats['avg_reward']:.3f}*\n"
        f"• Sharpe≈ *{stats['sharpe']:.2f}*\n"
        f"• Reject‑rate: *{stats['reject_rate']*100:.1f}%*\n"
        f"• Avg Dur: *{stats['avg_duration']:.1f} min*\n"
        f"• Avg PnL: *{stats['avg_pnl']:.2f} $*"
    )
    TG.send_message(txt)

    try:
        chart = _make_chart(rewards)
        TG.send_photo(chart, caption="Reward trend")
    except Exception as e:
        logger.error(f"[MetaReporter] chart send failed: {e}")

    _LAST_SENT_TS = time.time()