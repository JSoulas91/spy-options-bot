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

from meta.meta_state import build_meta_state_for_entry, build_meta_state_for_exit
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
        s_t = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices

def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    return f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"

def random_confidence() -> float:
    return round(np.random.beta(5, 2), 2)

def compute_indicators(prices: list[float], volumes: list[int], idx: int):
    window_20 = prices[max(0, idx - 19):idx + 1]
    volume_slice = volumes[max(0, idx - 19):idx + 1]
    close = prices[idx]

    vwap = np.average(window_20, weights=volume_slice) if window_20 else close
    ema_20 = sum(window_20) / len(window_20) if window_20 else close
    rsi_14 = 50 + RNG.uniform(-10, 10)  # placeholder

    return {
        "vwap": round(vwap, 2),
        "ema_20": round(ema_20, 2),
        "rsi_14": round(rsi_14, 2)
    }

def construct_bars(prices: list[float], volumes: list[int], interval: int):
    bars = []
    for i in range(0, len(prices), interval):
        chunk = prices[i:i+interval]
        vol_chunk = volumes[i:i+interval]
        if len(chunk) < interval:
            continue
        o, h, l, c = chunk[0], max(chunk), min(chunk), chunk[-1]
        v = sum(vol_chunk)
        bars.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars

def simulate_trade(day_idx: int, step_idx: int, prices: list[float], volumes: list[int], vix: float, macro_regime_shift: bool):
    minute_cutoff = 60 * 5 * 10  # ensure 10 1h bars
    if len(prices) < minute_cutoff:
        return None

    bars_1m = construct_bars(prices, volumes, 1)
    bars_5m = construct_bars(prices, volumes, 5)
    bars_15m = construct_bars(prices, volumes, 15)
    bars_1h = construct_bars(prices, volumes, 60)
    bars_1d = construct_bars(prices, volumes, len(prices))

    option_type = RNG.choice(["C", "P"])
    start_idx = RNG.randint(minute_cutoff, len(prices) - 241)

    is_swing = RNG.random() < 0.2
    duration = RNG.randint(10, 60 if not is_swing else 240)
    end_idx = start_idx + duration
    if end_idx >= len(prices):
        return None

    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    option_sym = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)

    confidence = random_confidence()
    hour = (start_idx // 60) % 24
    atr = RNG.uniform(2, 6)

    indicators = compute_indicators(prices, volumes, start_idx)
    bar_open = prices[start_idx]
    bar_high = round(bar_open * (1 + RNG.uniform(0, 0.001)), 2)
    bar_low = round(bar_open * (1 - RNG.uniform(0, 0.001)), 2)
    volume = volumes[start_idx]

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

    base_meta_state_dict = {
        "confidence": confidence,
        "vix": vix,
        "hour": hour,
        "is_swing": int(is_swing),
        "atr": atr,
        "trade_success_prob": trade_success_prob,
        "predicted_direction": predicted_direction,
        "entropy": entropy,
    }

    entry_meta_state = build_meta_state_for_entry(
        base_meta_state_dict,
        data_1m={"bars": bars_1m},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=confidence,
        trade_type=int(is_swing)
    )

    action = meta_agent.act(entry_meta_state)
    final_price = prices[end_idx]
    trade_result = (final_price - bar_open) if option_type == "C" else (bar_open - final_price)

    reward, shaped = compute_shaped_reward(trade_result, confidence)

    if shaped < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        return None

    exit_meta_state = build_meta_state_for_exit(
        base_meta_state_dict,
        data_1m={"bars": bars_1m[:end_idx+1]},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=confidence,
        trade_type=int(is_swing)
    )

    log_entry = {
        "timestamp": str(datetime.utcnow()),
        "day": day_idx,
        "trade_idx": step_idx,
        "option": option_sym,
        "strike": strike,
        "type": option_type,
        "open_price": bar_open,
        "final_price": final_price,
        "pnl": trade_result,
        "raw_reward": reward,
        "shaped_reward": shaped,
        "features": feature_dict,
        "classifier": {
            "prob": trade_success_prob,
            "direction": predicted_direction,
            "class_probabilities": class_probabilities,
            "entropy": entropy,
        },
        "meta_action": action,
        "meta_state_entry": entry_meta_state.tolist(),
        "meta_state_exit": exit_meta_state.tolist(),
    }

    log_training_example(feature_dict, trade_success_prob, predicted_direction, trade_result)
    return log_entry

def main():
    if META_LOG_PATH.exists():
        META_LOG_PATH.unlink()

    for day in range(SIM_DAYS):
        prices = gbm_path(390, START_PRICE, GBM_MU, GBM_SIGMA, 1/390)
        volumes = [RNG.randint(300_000, 1_000_000) for _ in prices]

        vix_spike = RNG.random() < 0.1
        macro_shift = RNG.random() < 0.2
        vix = RNG.uniform(22, 32) if vix_spike else RNG.uniform(14, 22)

        trades = []
        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, prices, volumes, vix, macro_shift)
            if log_entry:
                trades.append(log_entry)

        with open(META_LOG_PATH, "a") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        if (day + 1) % 50 == 0:
            logger.info(f"Simulated {day+1} days.")

    logger.info("Simulation complete.")
    send_telegram_message("✅ Simulation finished and saved to meta_log.jsonl")

if __name__ == "__main__":
    main()