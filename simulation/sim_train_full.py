import os
import json
import math
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from meta.meta_state import build_meta_state_for_entry
from meta.meta_agent import MetaAgent
from meta.reward_shaper import compute_shaped_reward
from utils.telegram_utils import send_telegram_message
from utils.logger import bot_logger as logger
from ml.logger import log_training_example
from ml.model_inference import ModelInference
from ml.feature_pipeline import build_features_for_trade

# ───────── simulation params ───────────────
SIM_DAYS = 500
TRADES_PER_DAY = 10
GBM_MU = 0.08
GBM_SIGMA = 0.22
START_PRICE = 450.0

META_LOG_PATH = Path("meta/meta_log.jsonl")
RNG = random.Random(42)
GARBAGE_KEEP_PROB = 0.05

meta_agent = MetaAgent()
model_inference = ModelInference()


def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    prices = [s0]
    for _ in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
        s_t = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt +
                                    sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    return f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"


def random_confidence() -> float:
    return round(np.random.beta(5, 2), 2)


def compute_indicators(prices: list[float], volumes: list[int], idx: int):
    window_20 = prices[max(0, idx - 19):idx + 1]
    close = prices[idx]
    volume = volumes[idx]

    vwap = np.average(window_20, weights=volumes[max(0, idx - 19):idx + 1]) if window_20 else close
    ema_20 = sum(window_20) / len(window_20) if window_20 else close
    rsi_14 = 50 + RNG.uniform(-10, 10)

    return {
        "vwap": round(vwap, 2),
        "ema_20": round(ema_20, 2),
        "rsi_14": round(rsi_14, 2)
    }


def simulate_trade(day_idx: int, step_idx: int, prices: list[float], volumes: list[int], vix: float):
    option_type = RNG.choice(["C", "P"])
    start_idx = RNG.randint(30, len(prices) - 31)
    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    option_sym = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)

    hour = RNG.randint(10, 15)
    confidence = random_confidence()
    atr = RNG.uniform(2, 6)

    base_meta_state_dict = {
        "confidence": confidence,
        "vix": vix,
        "hour": hour,
        "is_swing": 0,
        "atr": atr,
    }

    bar_open = prices[start_idx]
    bar_high = round(bar_open * (1 + RNG.uniform(0, 0.001)), 2)
    bar_low = round(bar_open * (1 - RNG.uniform(0, 0.001)), 2)
    volume = volumes[start_idx]
    indicators = compute_indicators(prices, volumes, start_idx)

    feature_dict = {
        "confidence": confidence,
        "hour": hour,
        "vix": vix,
        "atr": atr,
        "open": bar_open,
        "high": bar_high,
        "low": bar_low,
        "close": prices[start_idx],
        "volume": volume,
        "vwap": indicators["vwap"],
        "ema_20": indicators["ema_20"],
        "rsi_14": indicators["rsi_14"],
        "regime_bull": 1 if vix < 18 else 0,
        "regime_bear": 1 if vix >= 18 else 0,
    }

    features_df = build_features_for_trade(feature_dict)

    trade_success_prob = float(model_inference.predict_proba(features_df)[0])
    predicted_direction = int(model_inference.predict(features_df)[0])
    class_probabilities = {
        "success": trade_success_prob,
        "failure": 1 - trade_success_prob
    }
    entropy = -sum(p * math.log(p + 1e-9) for p in class_probabilities.values())

    base_meta_state_dict.update({
        "trade_success_prob": trade_success_prob,
        "predicted_direction": predicted_direction,
        "entropy": entropy,
    })

    meta_state = build_meta_state_for_entry(
        base_meta_state_dict,
        data_5m={}, data_15m={}, data_1h={}, data_1d={},
        confidence_score=confidence,
        trade_type=0
    )

    action_idx, agent_conf = meta_agent.select_action(meta_state)
    meta_agent.interpret_action(action_idx, agent_conf)

    fill_delay = RNG.randint(1, 5)
    fill_idx = min(start_idx + fill_delay, len(prices) - 1)
    fill_price = prices[fill_idx]

    slippage_pct = (fill_price - price_sig) / price_sig + RNG.gauss(0, 0.001)

    move_pct = RNG.uniform(-0.6, 0.6)
    raw_pnl_pct = move_pct * RNG.uniform(0.8, 1.2) * 0.3
    raw_pnl_pct = abs(raw_pnl_pct) if RNG.random() < confidence else -abs(raw_pnl_pct)
    raw_pnl_pct = max(min(raw_pnl_pct, 1.8), -0.9)

    fill_ratio = round(RNG.uniform(0.6, 1.0), 2)
    direction_outcome = 1 if raw_pnl_pct > 0 else 0

    trade = {
        "id": f"SIM.{day_idx}-{step_idx}",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "symbol": "SPY",
        "option_symbol": option_sym,
        "trade_type": 0,
        "confidence": confidence,
        "entry_price": fill_price,
        "slippage_pct": round(slippage_pct * 100, 3),
        "fill_delay_min": fill_delay,
        "fill_ratio": fill_ratio,
        "exit_price": round(fill_price * (1 + move_pct / 100), 2),
        "pnl": round(raw_pnl_pct * 100 * fill_ratio, 2),
        "meta_state": meta_state.tolist(),
        "bar": {
            "open": bar_open,
            "high": bar_high,
            "low": bar_low,
            "close": round(prices[fill_idx], 2),
            "volume": volume,
            **indicators
        },
        "meta_action": int(action_idx),
        "direction_outcome": direction_outcome,
        "classifier": {
            "trade_success_prob": round(trade_success_prob, 3),
            "predicted_direction": predicted_direction,
            "class_probabilities": class_probabilities,
            "entropy": round(entropy, 3),
            "features": features_df.iloc[0].to_dict()
        }
    }
    return trade


def append_meta_log(trade: dict, vix_val: float):
    META_LOG_PATH.parent.mkdir(exist_ok=True)
    shaped_reward = compute_shaped_reward({
        "trade": trade,
        "market": {"vix": vix_val},
        "exit_reason": "sim_exit"
    })

    signal_quality = (
        abs(shaped_reward) >= 0.3 and
        abs(trade["pnl"]) >= 1.5 and
        trade["confidence"] >= 0.3
    )

    if not signal_quality and RNG.random() > GARBAGE_KEEP_PROB:
        return

    payload = {
        "timestamp": trade["timestamp"],
        "trade": trade,
        "market": {"vix": vix_val},
        "exit_reason": "sim_exit",
        "meta_state": trade["meta_state"],
        "meta_action": trade["meta_action"],
        "reward": shaped_reward,
        "done": True,
        "high_quality": signal_quality,
    }

    with META_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


def summarize_sim_results():
    if not META_LOG_PATH.exists():
        return

    with META_LOG_PATH.open() as fh:
        lines = [json.loads(line) for line in fh if line.strip()]

    trades = [t["trade"] for t in lines]
    rewards = [t["reward"] for t in lines]
    win_trades = [t for t in trades if t["pnl"] > 0]

    summary = {
        "n_trades": len(trades),
        "avg_reward": round(np.mean(rewards), 3) if rewards else 0.0,
        "accuracy": round(len(win_trades) / len(trades), 3) if trades else 0.0,
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 2) if trades else 0.0
    }

    msg = (
        f"🧪 Sim complete: {summary['n_trades']} trades\n"
        f"✅ Accuracy: {summary['accuracy']*100:.1f}%\n"
        f"📈 Avg PnL: {summary['avg_pnl']}%\n"
        f"🎯 Avg Reward: {summary['avg_reward']}"
    )
    send_telegram_message(msg)


def simulate():
    logger.info("🧪 Starting synthetic back-test with realism …")
    current_price = START_PRICE

    for day in range(SIM_DAYS):
        logger.info(f"── Day {day + 1}/{SIM_DAYS}")

        minutes_per_day = 390
        prices = gbm_path(minutes_per_day, current_price,
                          GBM_MU / 252, GBM_SIGMA / math.sqrt(252), 1 / minutes_per_day)
        volumes = [RNG.randint(5000, 12000) for _ in range(minutes_per_day)]

        base_vix = 15 + RNG.gauss(0, 1.5)
        vix = max(12, min(base_vix, 30))
        current_price = prices[-1]

        for trade_idx in range(TRADES_PER_DAY):
            trade = simulate_trade(day, trade_idx, prices, volumes, vix)
            append_meta_log(trade, vix)

        if day % 10 == 0:
            meta_agent.save_model()

        if day % 50 == 0:
            send_telegram_message(f"Simulated {day} days of trades.")

        time.sleep(0.05)

    summarize_sim_results()


if __name__ == "__main__":
    simulate()