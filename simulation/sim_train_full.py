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
from meta.reward_shaper import RewardShaper
from utils.telegram_utils import send_telegram_message
from utils.logger import bot_logger as logger
from ml.logger import log_training_example
from ml.model_inference import ModelInference
from ml.feature_pipeline import build_features_for_trade

reward_shaper = RewardShaper()

# ───────── simulation params ───────────────
SIM_DAYS = 500
TRADES_PER_DAY = 12
GBM_MU = 0.08
GBM_SIGMA = 0.22
START_PRICE = 450.0

META_LOG_PATH = Path("meta/meta_log.jsonl")
RNG = random.Random(42)
GARBAGE_KEEP_PROB = 0.05
COMMISSION_PER_CONTRACT = 0.35
CONTRACT_MULTIPLIER = 100  # Options multiplier

meta_agent = MetaAgent()
model_inference = ModelInference()

print(f"SIM_DAYS={SIM_DAYS}, TRADES_PER_DAY={TRADES_PER_DAY}, START_PRICE={START_PRICE}")

def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    prices = [s0]
    for _ in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
        s_t = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    print(f"GBM path generated with {n_steps} steps from {s0} starting price.")
    print(f"First 5 prices: {prices[:5]}")
    print(f"Last 5 prices: {prices[-5:]}")
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    symbol = f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"
    print(f"Option symbol created: {symbol}")
    return symbol


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
    print(f"Constructed {len(bars)} bars from {len(prices)} prices with interval {interval}")
    if bars:
        print(f"First bar: {bars[0]}")
        print(f"Last bar: {bars[-1]}")
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

    print(f"Indicators at idx={idx}: {indicators}")
    return indicators


def black_scholes_price(s, k, t, r, sigma, call=True):
    if t <= 0:
        return max(0.0, s - k) if call else max(0.0, k - s)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    if call:
        return s * nd1 - k * math.exp(-r * t) * nd2
    else:
        return k * math.exp(-r * t) * (1 - nd2) - s * (1 - nd1)


def simulate_trade(day_idx, step_idx, prices, volumes, vix):
    # --- Step 1: Construct multi-timeframe bars ---
    bars_1m = construct_bars(prices, volumes, 1)
    bars_5m = construct_bars(prices, volumes, 5)
    bars_15m = construct_bars(prices, volumes, 15)
    bars_1h = construct_bars(prices, volumes, 60)
    bars_1d = construct_bars(prices, volumes, len(prices))  # 1-day bar from all prices

    # --- Step 2: Ensure minimum bar history ---
    if any(len(b) < 30 for b in [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        return None

    # --- Step 3: Force a dummy trade ---
    return {
        "timestamp": f"day_{day_idx}_step_{step_idx}",
        "pnl": RNG.uniform(-1, 1),
        "duration": RNG.randint(1, 100),
        "meta_state_entry": [0.5] * 100,
        "meta_state_exit": [0.5] * 100,
        "classifier_features": [0.1] * 20,
        "trade_type": "call"
    }


def main():
    # Removed file deletion to preserve existing logs

    for day in range(SIM_DAYS):
        vix_shift = RNG.uniform(14, 28)
        if RNG.random() < 0.08:
            vix_shift += RNG.uniform(5, 15)

        prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1/390)
        volumes = [RNG.randint(300_000, 1_000_000) for _ in prices]

        trades = []
        logger.debug(f"Day {day+1}: Starting simulation with VIX shift {vix_shift:.2f}")
        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, prices, volumes, vix_shift)
            if log_entry:
                trades.append(log_entry)
                logger.debug(f"Trade {trade_idx+1} generated: PnL={log_entry['pnl']}, duration={log_entry['duration']}")

        if trades:
            with open(META_LOG_PATH, "a") as f:
                for t in trades:
                    f.write(json.dumps(t) + "\n")
            logger.debug(f"Day {day+1}: Logged {len(trades)} trades to {META_LOG_PATH}")
        else:
            logger.debug(f"Day {day+1}: No trades generated")

        if (day + 1) % 50 == 0:
            logger.info(f"Simulated {day + 1} days.")

    logger.info("✅ Simulation complete.")
    send_telegram_message("✅ Simulation finished and saved to meta/meta_log.jsonl")


if __name__ == "__main__":
    main()
