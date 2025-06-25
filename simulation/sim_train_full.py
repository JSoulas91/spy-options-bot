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
WARM_UP_DAYS = 7

ACCUMULATED_1M_BARS = []
ACCUMULATED_VOLUMES = []

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


def construct_bars(prices, volumes, interval, start_time=None):
    if start_time is None:
        # Simulate a market open time if not provided
        start_time = datetime(2025, 1, 1, 9, 30)

    bars = []
    for i in range(0, len(prices) - interval + 1, interval):
        chunk = prices[i:i + interval]
        vol_chunk = volumes[i:i + interval]
        bar_time = start_time + timedelta(minutes=i)

        bars.append({
            "timestamp": bar_time,
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

def clean_bars(bars):
    return [bar for bar in bars if all(isinstance(bar.get(k), (int, float)) for k in ["open", "high", "low", "close", "volume"])]
    
def simulate_trade(day, trade_idx, bars_1m, volumes_1m, vix_shift):
    trade_minute = step_idx * 5 + 1  # Current minute (exclusive)

    # Define raw data required for multi-timeframe indicators
    required_lookback = {
        "1m": 60,
        "5m": 150,   # 5 * 30 bars of 5-min = 150 mins
        "15m": 300,  # 15 * 20 bars
        "1h": 600,   # 60 * 10 bars
        "1d": 1950,  # 390 * 5 bars (5 days)
    }

    max_lookback = max(required_lookback.values())
    if trade_minute < max_lookback:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: insufficient lookback for multi-timeframe bars")
        return None

    # Time offset logic
    trade_minutes_offset = step_idx * 5
    total_offset = timedelta(days=day_idx, minutes=trade_minutes_offset)
    base_time = datetime(2025, 1, 1, 9, 30) + total_offset

    # Build multi-timeframe bars
    bars_1m = construct_bars(prices[bar_end - 60:bar_end], volumes[bar_end - 60:bar_end], 1, start_time=base_time - timedelta(minutes=59))
    bars_5m = construct_bars(prices[bar_end - 150:bar_end], volumes[bar_end - 150:bar_end], 5, start_time=base_time - timedelta(minutes=145))
    bars_15m = construct_bars(prices[bar_end - 300:bar_end], volumes[bar_end - 300:bar_end], 15, start_time=base_time - timedelta(minutes=285))
    bars_1h = construct_bars(prices[bar_end - 600:bar_end], volumes[bar_end - 600:bar_end], 60, start_time=base_time - timedelta(minutes=540))
    bars_1d = construct_bars(prices[bar_end - 1950:bar_end], volumes[bar_end - 1950:bar_end], 390, start_time=base_time - timedelta(days=4, minutes=30))

  # 🧹 Clean bars before validation
    bars_1m = clean_bars(bars_1m)
    bars_5m = clean_bars(bars_5m)
    bars_15m = clean_bars(bars_15m)
    bars_1h = clean_bars(bars_1h)
    bars_1d = clean_bars(bars_1d)
    
    # Validate bars
    required_bars = {"1m": 60, "5m": 30, "15m": 20, "1h": 10, "1d": 5}
    for tf, bars in zip(required_bars, [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        if not isinstance(bars, list) or len(bars) < required_bars[tf]:
            logger.debug(f"Skipping trade {step_idx} on day {day_idx}: insufficient bars for {tf}")
            return None

    max_start_idx = len(bars_1m) - 60
    if max_start_idx <= required_bars["1m"]:
        return None

    start_idx = RNG.randint(required_bars["1m"], max_start_idx)
    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    option_type = RNG.choice(["C", "P"])
    expiry_days = RNG.randint(7, 30)
    t_expiry = expiry_days / 365

    option_price = black_scholes_price(
        s=price_sig,
        k=strike,
        t=t_expiry,
        r=0.01,
        sigma=0.25,
        call=(option_type == "C")
    )

    slippage = RNG.uniform(-0.5, 0.5) / 100
    fill_pct = RNG.uniform(0.7, 1.0)
    entry_price = round(option_price * (1 + slippage), 2)

    option_sym = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)
    is_swing = RNG.random() < 0.2  # random swing trade decision

    # --- Compute indicators from bars_1m
    indicators = compute_all_indicators(bars_1m, volumes, len(bars_1m) - 1)

    # --- Classifier features ---
    classifier_confidence = round(np.random.beta(5, 2), 2)
    setup_quality = round(RNG.uniform(0.6, 1.0), 2)
    vix = round(RNG.uniform(15, 35), 2)
    realized_vol = round(np.std(prices[start_idx - 20:start_idx]), 2) if start_idx >= 20 else 1.5
    trade_type = 0 if option_type == "C" else 1
    total_signals_today = RNG.randint(0, 10)

    classifier_features = {
        'confidence': classifier_confidence,
        'setup_quality': setup_quality,
        'vix': vix,
        'realized_vol': realized_vol,
        'trade_type': trade_type,
        'total_signals_today': total_signals_today,
        'ema_20': indicators['ema_20'],
        'rsi_14': indicators['rsi_14'],
        'macd': indicators['macd'],
        'macd_signal': indicators['macd_signal'],
        'macd_hist': indicators['macd_hist'],
        'bb_upper': indicators['bb_upper'],
        'bb_middle': indicators['bb_middle'],
        'bb_lower': indicators['bb_lower'],
        'vwap': indicators['vwap'],
        'atr_14': indicators['atr_14'],
        'adx_14': indicators['adx_14'],
    }

    features_df = build_features_for_trade(classifier_features)
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame([classifier_features])
    if features_df.shape[0] != 1:
        return None

    try:
        class_probs = classifier.predict_proba(features_df)[0]
        predicted_direction = int(np.argmax(class_probs))
        trade_success_prob = float(class_probs[predicted_direction])
    except Exception as e:
        logger.debug(f"Classifier prediction failed: {e}")
        return None

    class_probabilities = {
        "success": trade_success_prob,
        "failure": 1 - trade_success_prob
    }
    entropy = -sum(p * math.log(p + 1e-9) for p in class_probabilities.values())

    meta_entry = build_meta_state_for_entry(
        data_1m={"bars": bars_1m},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=classifier_confidence,
        trade_type=int(is_swing),
        classifier_output={
            "trade_success_prob": trade_success_prob,
            "predicted_direction": predicted_direction,
            "class_probabilities": class_probabilities,
            "entropy": entropy
        }
    )
    if meta_entry is None:
        return None

    action, agent_confidence = meta_agent.select_action(meta_entry)
    if action == 0:
        return None

    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    if start_idx + duration >= len(prices):
        return None
    final_price = prices[start_idx + duration]

    minutes_per_year = 252 * 6.5 * 60
    time_left = max(t_expiry - (duration * 1) / minutes_per_year, 0.01)
    new_option_price = black_scholes_price(
        s=final_price,
        k=strike,
        t=time_left,
        r=0.01,
        sigma=0.25,
        call=(option_type == "C")
    )
    exit_price = round(new_option_price * (1 + slippage), 2)

    gross_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * fill_pct
    total_commission = 2 * COMMISSION_PER_CONTRACT
    raw_pnl = gross_pnl - total_commission
    initial_cost = entry_price * CONTRACT_MULTIPLIER * fill_pct + 1e-9
    pct_pnl = (raw_pnl / initial_cost) * 100
    trade_result = pct_pnl
    atr = indicators.get('atr_14', 1.0)

    meta_exit = build_meta_state_for_exit(
        data_1m={"bars": bars_1m},
        data_5m={"bars": bars_5m},
        data_15m={"bars": bars_15m},
        data_1h={"bars": bars_1h},
        data_1d={"bars": bars_1d},
        confidence_score=agent_confidence,
        trade_type=int(is_swing),
    )
    if meta_exit is None:
        return None

    direction_correct = (
        (predicted_direction == 1 and final_price > price_sig) or
        (predicted_direction == 0 and final_price < price_sig)
    )

    shaped_reward = reward_shaper.compute_shaped_reward(
        trade_result={
            "pct_pnl": trade_result,
            "setup_quality": setup_quality,
            "entry_quality": abs(trade_result) / atr,
            "direction_correct": direction_correct,
            "trades_today": step_idx,
            "was_successful": trade_result > 0,
            "risk_reward_ratio": abs(trade_result) / atr,
            "time_to_target": duration,
            "max_drawdown": RNG.uniform(0, abs(trade_result) * 0.3),
            "exploration_bonus": RNG.uniform(0, 0.2),
            "skipped_strong_signal": RNG.random() < 0.05,
        },
        classifier_output={
            "confidence": classifier_confidence,
            "entropy": entropy
        },
        regime="neutral",
        agent_confidence=agent_confidence,
    )

    if shaped_reward < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        return None

    entry_bar = bars_1m[start_idx]
    ts = entry_bar["timestamp"]
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts)

    log_training_example(
        timestamp=ts,
        close=entry_bar["close"],
        features=classifier_features,
        label=trade_result
    )

    return {
        "timestamp": str(ts),
        "day": day_idx,
        "trade_idx": step_idx,
        "option": option_sym,
        "strike": strike,
        "type": option_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "final_price": final_price,
        "fill_pct": round(fill_pct, 2),
        "duration": duration,
        "pct_pnl": round(trade_result, 2),
        "shaped_reward": shaped_reward,
        "features": classifier_features,
        "classifier": {
            "prob": trade_success_prob,
            "direction": predicted_direction,
            "class_probabilities": class_probabilities,
            "entropy": entropy,
        },
        "meta_action": action,
        "agent_confidence": agent_confidence,
        "entry_state": meta_entry.tolist(),
        "exit_state": meta_exit.tolist(),
    }

def main():
    global ACCUMULATED_1M_BARS, ACCUMULATED_VOLUMES

    for day in range(SIM_DAYS):
        # Skip first 7 days to allow multi-timeframe bars (1h, 4h, etc.) to build up
        if day < WARM_UP_DAYS:
            logger.debug(f"⏩ Skipping Day {day+1}: Warming up multi-timeframe history")
            daily_prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
            daily_volumes = [RNG.randint(300_000, 1_000_000) for _ in daily_prices]
            ACCUMULATED_1M_BARS.extend(daily_prices)
            ACCUMULATED_VOLUMES.extend(daily_volumes)
            continue

        vix_shift = RNG.uniform(14, 28)
        if RNG.random() < 0.08:
            vix_shift += RNG.uniform(5, 15)

        daily_prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
        daily_volumes = [RNG.randint(300_000, 1_000_000) for _ in daily_prices]

        ACCUMULATED_1M_BARS.extend(daily_prices)
        ACCUMULATED_VOLUMES.extend(daily_volumes)

        trades = []
        logger.debug(f"Day {day+1}: Starting simulation with VIX shift {vix_shift:.2f}")
        successful_trades = 0

        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, ACCUMULATED_1M_BARS, ACCUMULATED_VOLUMES, vix_shift)
            if log_entry:
                trades.append(log_entry)
                successful_trades += 1
                logger.debug(f"✅ Trade {trade_idx+1} generated: PnL={log_entry['pct_pnl']}, duration={log_entry['duration']}")
            else:
                logger.debug(f"❌ Trade {trade_idx+1} skipped or failed (simulate_trade returned None)")

        if trades:
            with open(META_LOG_PATH, "a") as f:
                for t in trades:
                    f.write(json.dumps(t) + "\n")
            logger.debug(f"Day {day+1}: Logged {len(trades)} trades to {META_LOG_PATH}")
        else:
            logger.debug(f"Day {day+1}: No trades generated")

        logger.info(f"Day {day+1}: {successful_trades}/{TRADES_PER_DAY} trades returned from simulate_trade()")

        if (day + 1) % 50 == 0:
            logger.info(f"Simulated {day + 1} days.")

    logger.info("✅ Simulation complete.")
    send_telegram_message("✅ Simulation finished and saved to meta/meta_log.jsonl")


if __name__ == "__main__":
    main()