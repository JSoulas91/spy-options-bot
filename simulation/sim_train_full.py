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


def simulate_trade(day_idx, step_idx, prices, volumes, vix):
    if len(prices) < 1800:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: insufficient raw price history")
        return None

    # 12 trades/day → every 5 minutes
    trade_minutes_offset = step_idx * 5
    total_offset = timedelta(days=day_idx, minutes=trade_minutes_offset)
    base_time = datetime(2025, 1, 1, 9, 30) + total_offset

    bars_1m = construct_bars(prices, volumes, 1, start_time=base_time)
    bars_5m = construct_bars(prices, volumes, 5, start_time=base_time)
    bars_15m = construct_bars(prices, volumes, 15, start_time=base_time)
    bars_1h = construct_bars(prices, volumes, 60, start_time=base_time)
    bars_1d = construct_bars(prices, volumes, 390, start_time=base_time)
    for tf, bars in zip(["1m", "5m", "15m", "1h", "1d"], [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        if not isinstance(bars, pd.DataFrame):
            print(f"[ERROR] bars_{tf} is not a DataFrame")
        elif bars.empty:
            print(f"[WARNING] bars_{tf} is empty")
    
    if len(bars_1m) < 60 or len(bars_5m) < 30 or len(bars_15m) < 20 or len(bars_1h) < 10 or len(bars_1d) < 5:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: insufficient multi-timeframe bars")
        return None

    # Convert lists to DataFrames for meta-state functions that require .iloc
    bars_1m = pd.DataFrame(bars_1m)
    bars_5m = pd.DataFrame(bars_5m)
    bars_15m = pd.DataFrame(bars_15m)
    bars_1h = pd.DataFrame(bars_1h)
    bars_1d = pd.DataFrame(bars_1d)
    
    option_type = RNG.choice(["C", "P"])
    
    min_bars_needed = {
        "1m": 60,
        "5m": 30,
        "15m": 20,
        "1h": 10,
        "1d": 5
    }
    required_lookback = max(
        min_bars_needed["1m"],
        min_bars_needed["5m"] * 5,
        min_bars_needed["15m"] * 15,
        min_bars_needed["1h"] * 60,
        min_bars_needed["1d"] * 390
    )
    start_idx = RNG.randint(required_lookback, len(prices) - 61)
    entry_bar = bars_1m[start_idx]
    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)

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

    classifier_confidence = round(np.random.beta(5, 2), 2)
    hour = RNG.randint(10, 15)
    atr = RNG.uniform(2, 6)
    is_swing = RNG.random() < 0.25

    indicators = compute_all_indicators(prices, volumes, start_idx)
    setup_quality = RNG.uniform(0.6, 1.0)

    classifier_features = {
        "confidence": classifier_confidence,
        "setup_quality": setup_quality,
        "vix": vix,
        "realized_vol": RNG.uniform(0.1, 0.6),
        "trade_type": int(is_swing),
        "total_signals_today": RNG.randint(1, 7),
        **indicators
    }

    features_df = build_features_for_trade(classifier_features)

    try:
        trade_success_prob = float(model_inference.predict_proba(features_df)[0])
        predicted_direction = int(model_inference.predict(features_df)[0])
    except Exception as e:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: classifier prediction failed ({e})")
        return None

    class_probabilities = {
        "success": trade_success_prob,
        "failure": 1 - trade_success_prob
    }
    entropy = -sum(p * math.log(p + 1e-9) for p in class_probabilities.values())

    base_meta_state_dict = {
        "confidence": classifier_confidence,
        "vix": vix,
        "hour": hour,
        "is_swing": int(is_swing),
        "atr": atr,
        "trade_success_prob": trade_success_prob,
        "predicted_direction": predicted_direction,
        "entropy": entropy,
    }
    # Check bars type to debug iloc error
    print("DEBUG: Checking types of bars before building meta state for entry:")
    print(f"  bars_1m: {type(bars_1m)}")
    print(f"  bars_5m: {type(bars_5m)}")
    print(f"  bars_15m: {type(bars_15m)}")
    print(f"  bars_1h: {type(bars_1h)}")
    print(f"  bars_1d: {type(bars_1d)}")
    
    if not isinstance(bars_1m, pd.DataFrame):
        print(f"  WARNING: bars_1m is NOT a DataFrame! Sample: {bars_1m[:3]}")
    
    if not isinstance(bars_5m, pd.DataFrame):
        print(f"  WARNING: bars_5m is NOT a DataFrame! Sample: {bars_5m[:3]}")
    
    if not isinstance(bars_15m, pd.DataFrame):
        print(f"  WARNING: bars_15m is NOT a DataFrame! Sample: {bars_15m[:3]}")
    
    if not isinstance(bars_1h, pd.DataFrame):
        print(f"  WARNING: bars_1h is NOT a DataFrame! Sample: {bars_1h[:3]}")
    
    if not isinstance(bars_1d, pd.DataFrame):
        print(f"  WARNING: bars_1d is NOT a DataFrame! Sample: {bars_1d[:3]}")

    meta_entry = build_meta_state_for_entry(
        data_1m=bars_1m,
        data_5m=bars_5m,
        data_15m=bars_15m,
        data_1h=bars_1h,
        data_1d=bars_1d,
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
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: entry meta-state construction failed")
        return None
        
    action, agent_confidence = meta_agent.select_action(meta_entry)
    meta_info = {}  # or some placeholder if needed downstream
    logger.debug(f"[MetaAgent] Action {action}, Confidence {agent_confidence:.2f}, Details: {meta_info}")

    if action == 0:
        return None  # Meta-agent says no trade

    # In simulate_trade(), after creating meta_entry:
    action_idx = action  # From earlier select_action() call
    agent_conf = agent_confidence  # From earlier select_action() call
    action_details = meta_agent.interpret_action(action_idx, agent_conf)

    # Optionally log:
    logger.debug(f"[MetaAgent] Action {action_idx}, Confidence {agent_conf:.2f}, Details: {action_details}")

    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    if start_idx + duration >= len(prices):
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: trade duration {duration} exceeds available price data")
        return None

    final_price = prices[start_idx + duration]
    new_option_price = black_scholes_price(
        s=final_price,
        k=strike,
        t=max(t_expiry - duration / 390 / 6.5 / 252, 0.01),
        r=0.01,
        sigma=0.25,
        call=(option_type == "C")
    )

    exit_price = round(new_option_price * (1 + slippage), 2)
    gross_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * fill_pct
    total_commission = 2 * COMMISSION_PER_CONTRACT
    raw_pnl = gross_pnl - total_commission

    # ✅ Convert raw PnL to percentage return
    initial_cost = entry_price * CONTRACT_MULTIPLIER * fill_pct + 1e-6  # avoid div by zero
    pct_pnl = (raw_pnl / initial_cost) * 100  # percentage return
    trade_result = pct_pnl
    
    meta_exit = build_meta_state_for_exit(
        data_1m=bars_1m,
        data_5m=bars_5m,
        data_15m=bars_15m,
        data_1h=bars_1h,
        data_1d=bars_1d,
        confidence_score=agent_confidence,
        trade_type=int(is_swing),
    )
    if meta_exit is None:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: exit meta-state construction failed")
        return None

    direction_correct = (
        (predicted_direction == "long" and final_price > entry_price) or
        (predicted_direction == "short" and final_price < entry_price)
    )
    
    trades_today = step_idx  # Number of trades so far in the current day
    was_successful = trade_result > 0
    risk_reward_ratio = abs(trade_result) / atr if atr > 0 else 1.0
    time_to_target = duration
    max_drawdown = RNG.uniform(0, abs(trade_result) * 0.3)
    exploration_bonus = RNG.uniform(0, 0.2)
    skipped_strong_signal = RNG.random() < 0.05
    regime = "neutral"
    
    shaped_reward = reward_shaper.compute_shaped_reward(
        trade_result={
            "pct_pnl": trade_result,
            "setup_quality": setup_quality,
            "entry_quality": abs(trade_result) / atr,
            "direction_correct": direction_correct,  # e.g. predicted vs actual direction
            "trades_today": trades_today,
            "was_successful": was_successful,
            "risk_reward_ratio": risk_reward_ratio,
            "time_to_target": time_to_target,
            "max_drawdown": max_drawdown,
            "exploration_bonus": exploration_bonus,
            "skipped_strong_signal": skipped_strong_signal,
        # Add any other needed keys from the reward function
        },
        classifier_output={
            "confidence": classifier_confidence,
            "entropy": entropy
        },
        regime=regime,
        agent_confidence=agent_confidence,
    )

    if shaped_reward < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        logger.debug(f"Skipping trade {step_idx} on day {day_idx}: shaped reward {shaped:.2f} below threshold")
        return None
    
    # Convert timestamp if needed before passing
    entry_bar = bars_1m[start_idx]
    ts = entry_bar["timestamp"]
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts)

    close_val = entry_bar["close"]

    print(f"Timestamp type before logging: {type(ts)}")
    print(f"Timestamp value before logging: {ts}")

    log_training_example(
        timestamp=ts,
        close=entry_bar['close'],
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
    # Removed file deletion to preserve existing logs

    for day in range(SIM_DAYS):
        vix_shift = RNG.uniform(14, 28)
        if RNG.random() < 0.08:
            vix_shift += RNG.uniform(5, 15)

        prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
        volumes = [RNG.randint(300_000, 1_000_000) for _ in prices]

        trades = []
        logger.debug(f"Day {day+1}: Starting simulation with VIX shift {vix_shift:.2f}")
        successful_trades = 0

        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, prices, volumes, vix_shift)
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