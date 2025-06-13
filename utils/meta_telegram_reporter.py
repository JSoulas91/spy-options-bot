import os
import io
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from utils.telegram_utils import send_telegram_message

LOG_PATH = "meta/meta_log.jsonl"
MAX_EPISODES = 100  # Limit to recent episodes for plotting

def load_meta_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def generate_summary(log_data):
    df = pd.DataFrame(log_data)
    if df.empty:
        return "No data available."

    df["reward"] = df["reward"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    avg_reward = df["reward"].mean()
    max_reward = df["reward"].max()
    min_reward = df["reward"].min()
    last_reward = df["reward"].iloc[-1]
    n_trades = len(df)

    regime_counts = df["regime"].value_counts().to_dict()
    action_counts = df["action"].value_counts().to_dict()

    summary = f"""
📊 *Meta-Agent Training Summary*
🗓️ Period: {df['timestamp'].min().strftime('%Y-%m-%d')} → {df['timestamp'].max().strftime('%Y-%m-%d')}
📈 Trades: {n_trades}
💡 Reward Avg: `{avg_reward:.2f}`, Max: `{max_reward:.2f}`, Min: `{min_reward:.2f}`, Last: `{last_reward:.2f}`
🧠 Regimes Seen: {regime_counts}
🎯 Actions Taken: {action_counts}
"""
    return summary

def generate_reward_plot(log_data):
    df = pd.DataFrame(log_data)
    if df.empty or "reward" not in df:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["reward"] = df["reward"].astype(float)

    # Keep only the last MAX_EPISODES
    df = df.tail(MAX_EPISODES)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["timestamp"], df["reward"], marker="o", linestyle="-", color="teal", label="Reward")
    ax.set_title("Meta-Agent Reward Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def send_meta_agent_report():
    log_data = load_meta_log()
    if not log_data:
        send_telegram_message("⚠️ No meta-agent log data found.")
        return

    summary = generate_summary(log_data)
    image_bytes = generate_reward_plot(log_data)

    send_telegram_message(summary.strip(), image_bytes=image_bytes)

if __name__ == "__main__":
    send_meta_agent_report()