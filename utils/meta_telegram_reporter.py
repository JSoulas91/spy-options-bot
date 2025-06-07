# utils/meta_telegram_reporter.py
"""
Send concise meta‑agent training summaries *plus* a reward‑curve image to Telegram.
"""
from __future__ import annotations
import io, time, requests, matplotlib.pyplot as plt
from typing import List, Dict

from utils.logger         import bot_logger as logger
from utils.telegram_utils import send_telegram_message
from config               import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

LAST_SENT         = 0.0
MIN_INTERVAL_SEC  = 60                # throttle updates

# ───────────────────────────────── plotting helper
def _make_reward_plot(reward_history: List[float]) -> bytes:
    """Return PNG bytes of a simple reward line chart."""
    fig, ax = plt.subplots()
    ax.plot(reward_history, linewidth=2)
    ax.set_title("Meta‑Agent Avg Reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _send_photo(img_bytes: bytes, caption: str = ""):
    """Low‑level Telegram `sendPhoto` helper."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("reward.png", img_bytes)}
    data  = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=data, files=files, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[Tele‑photo] {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"[Tele‑photo] {e}")

# ───────────────────────────────── public API
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
    """
    global LAST_SENT
    if time.time() - LAST_SENT < MIN_INTERVAL_SEC:
        return  # avoid Telegram spam

    txt = (
        f"📚 *Meta Training* — Epoch *{stats['epoch']}*\n"
        f"Avg R: `{stats['avg_reward']:.4f}`  σR: `{stats['reward_std']:.4f}`\n"
        f"Sharpe≈ `{stats['sharpe']:.2f}`  Reject: `{stats['reject_rate']*100:.1f}%`\n"
        f"Dur≈ `{stats['avg_duration']:.1f}` min   PnL≈ `${stats['avg_pnl']:.2f}`"
    )
    send_telegram_message(txt)

    # attach chart
    try:
        png = _make_reward_plot(reward_history)
        _send_photo(png, caption="Reward trend")
    except Exception as e:
        logger.error(f"[MetaReport] plot failed: {e}")

    LAST_SENT = time.time()