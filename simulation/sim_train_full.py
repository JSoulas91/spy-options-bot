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
TRADES_PER_DAY = 12
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
    for i in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
        s_t = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    return f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"


def construct_bars(prices, volumes, interval):
    bars = []
    for i in range(0, len(prices) - interval + 1, interval):
        chunk = prices[i:i+interval]
        vol_chunk = volumes[i:i+interval]
        if len(chunk) < interval:
            continue
        bars.append({
            "open": chunk[0],
            "high": max(chunk),
            "low": min(chunk),
            "close": chunk[-1],
            "volume": sum(vol_chunk)
        })
    return bars


def compute_all_indicators(prices, volumes, idx):
    window = prices[max(0, idx - 50):idx + 1]
    closes = pd.Series(window)

    indicators = {}

    indicators["ema_20"] = closes.ewm(span=20).mean().iloc[-1]
    delta = closes.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean().iloc[-1]
    avg_loss = down.rolling(window=14).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-6)
    indicators["rsi_14"] = 100 - (100 / (1 + rs))

    exp1 = closes.ewm(span=12, adjust=False).mean()
    exp2 = closes.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    indicators["macd"] = macd.iloc[-1]
    indicators["macd_signal"] = signal.iloc[-1]
    indicators["macd_hist"] = (macd - signal).iloc[-1]

    std = closes.rolling(window=20).std().iloc[-1]
    middle = closes.rolling(window=20).mean().iloc[-1]
    indicators["bb_middle"] = middle
    indicators["bb_upper"] = middle + 2 * std
    indicators["bb_lower"] = middle - 2 * std

    vwap = np.average(window, weights=volumes[max(0, idx - 50):idx + 1])
    indicators["vwap"] = vwap

    tr = pd.Series([max(closes.iloc[i] - closes.iloc[i-1], 0) for i in range(1, len(closes))])
    indicators["atr_14"] = tr.rolling(window=14).mean().iloc[-1]

    adx = RNG.uniform(10, 35)
    indicators["adx_14"] = adx

    for k in indicators:
        indicators[k] = round(float(indicators[k]), 4)

    return indicators


def simulate_trade(day_idx, step_idx, prices, volumes, vix):
    minute_cutoff = 60 * 5 * 10
    if len(prices) < minute_cutoff:
        return None

    bars_1m = construct_bars(prices, volumes, 1)
    bars_5m = construct_bars(prices, volumes, 5)
    bars_15m = construct_bars(prices, volumes, 15)
    bars_1h = construct_bars(prices, volumes, 60)
    bars_1d = construct_bars(prices, volumes, len(prices))

    option_type = RNG.choice(["C", "P"])
    start_idx = RNG.randint(minute_cutoff, len(prices) - 61)
    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    option_sym = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)

    confidence = round(np.random.beta(5, 2), 2)
    hour = RNG.randint(10, 15)
    atr = RNG.uniform(2, 6)
    is_swing = RNG.random() < 0.25

    indicators = compute_all_indicators(prices, volumes, start_idx)

    feature_dict = {
        "confidence": confidence,
        "setup_quality": RNG.uniform(0.6, 1.0),
        "vix": vix,
        "realized_vol": RNG.uniform(0.1, 0.6),
        "trade_type": int(is_swing),
        "total_signals_today": RNG.randint(1, 7),
        **indicators
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

    meta_entry = build_meta_state_for_entry(
        base_meta_state_dict,
        data_1m={"bars": bars_1m},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=confidence,
        trade_type=int(is_swing),
    )

    if meta_entry is None:
        return None

    action = meta_agent.act(meta_entry)
    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    if start_idx + duration >= len(prices):
        return None

    final_price = prices[start_idx + duration]
    trade_result = (final_price - price_sig) if option_type == "C" else (price_sig - final_price)
    reward, shaped = compute_shaped_reward(trade_result, confidence)

    if shaped < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        return None

    meta_exit = build_meta_state_for_exit(
        base_meta_state_dict,
        data_1m={"bars": bars_1m},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=confidence,
        trade_type=int(is_swing),
    )

    if meta_exit is None:
        return None

    log_training_example(feature_dict, trade_success_prob, predicted_direction, trade_result)

    return {
        "timestamp": str(datetime.utcnow()),
        "day": day_idx,
        "trade_idx": step_idx,
        "option": option_sym,
        "strike": strike,
        "type": option_type,
        "open_price": price_sig,
        "final_price": final_price,
        "duration": duration,
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
        "entry_state": meta_entry.tolist(),
        "exit_state": meta_exit.tolist(),
    }


def main():
    if META_LOG_PATH.exists():
        META_LOG_PATH.unlink()

    for day in range(SIM_DAYS):
        vix_shift = RNG.uniform(14, 28)
        if RNG.random() < 0.08:
            vix_shift += RNG.uniform(5, 15)

        prices = gbm_path(390, START_PRICE, GBM_MU, GBM_SIGMA, 1/390)
        volumes = [RNG.randint(300_000, 1_000_000) for _ in prices]

        trades = []
        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, prices, volumes, vix_shift)
            if log_entry:
                trades.append(log_entry)

        with open(META_LOG_PATH, "a") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        if (day + 1) % 50 == 0:
            logger.info(f"Simulated {day + 1} days.")

    logger.info("✅ Simulation complete.")
    send_telegram_message("✅ Simulation finished and saved to meta/meta_log.jsonl")


if __name__ == "__main__":
    main()