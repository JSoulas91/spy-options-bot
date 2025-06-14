import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from utils.telegram_utils import send_plot, send_telegram_message
from config import META_LOG_PATH

def load_log_as_dataframe():
    if not os.path.exists(META_LOG_PATH):
        return pd.DataFrame()

    with open(META_LOG_PATH) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    if not lines:
        return pd.DataFrame()

    df = pd.DataFrame(lines)
    if "timestamp" not in df.columns:
        return pd.DataFrame()  # ❌ skip if no timestamp info

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp")

def generate_summary(df: pd.DataFrame):
    if df.empty:
        return "No valid meta-agent logs available."

    last_3d = df[df["timestamp"] >= datetime.now() - timedelta(days=3)]
    avg_r = last_3d["reward"].mean() if "reward" in last_3d.columns else None
    count = len(last_3d)

    return (
        f"📊 Meta-Agent Report\n"
        f"Records (last 3d): {count}\n"
        f"Avg Reward: {avg_r:.4f}" if avg_r is not None else "No recent reward data."
    )

def send_meta_agent_report():
    df = load_log_as_dataframe()
    summary = generate_summary(df)
    send_telegram_message(summary)

    if not df.empty and "reward" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["reward"], label="Reward")
        plt.title("Meta-Agent Reward Trend")
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        send_plot(plt)
        plt.close()

def send_training_report(stats: dict, history: list[float]):
    msg = (
        f"📈 PPO Epoch {stats['epoch']}\n"
        f"Avg Reward: {stats['avg_reward']:.4f}\n"
        f"Reward Std: {stats['reward_std']:.4f}\n"
        f"Entropy Coef: {stats.get('entropy_coef', 0):.6f}\n"
        f"Learning Rate: {stats.get('learning_rate', 0):.8f}"
    )
    send_telegram_message(msg)

    if history:
        plt.figure(figsize=(10, 3))
        plt.plot(history, label="Avg Reward")
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        send_plot(plt)
        plt.close()