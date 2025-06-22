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

        bars.append({
            "open": chunk[0],
            "high": max(chunk),
            "low": min(chunk),
            "close": chunk[-1],
            "volume": sum(vol_chunk)
        })
    
    return bars  # always return list, empty if no bars


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
    # Ensure we have enough raw data to build all bars
    if len(prices) < 1800:
        return None

    # Construct multi-timeframe bars
    bars_1m = construct_bars(prices, volumes, 1)
    bars_5m = construct_bars(prices, volumes, 5)
    bars_15m = construct_bars(prices, volumes, 15)
    bars_1h = construct_bars(prices, volumes, 60)
    bars_1d = construct_bars(prices, volumes, len(prices))  # Daily bar (1 bar)

    # Ensure each bar set has enough history for meta-state windows
    if any(len(b) < 30 for b in [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        return None

    print(f"Bars lengths — 1m: {len(bars_1m)}, 5m: {len(bars_5m)}, 15m: {len(bars_15m)}, 1h: {len(bars_1h)}, 1d: {len(bars_1d)}")

    option_type = RNG.choice(["C", "P"])
    start_idx = RNG.randint(300, len(prices) - 61)
    price_sig = prices[start_idx]
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    print(f"Option type: {option_type}, start_idx: {start_idx}, price_sig: {price_sig}, strike: {strike}")

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
    print(f"Expiry days: {expiry_days}, t_expiry: {t_expiry}, option_price: {option_price}")

    slippage = RNG.uniform(-0.5, 0.5) / 100
    fill_pct = RNG.uniform(0.7, 1.0)
    print(f"Slippage: {slippage:.5f}, Fill percentage: {fill_pct:.2f}")

    entry_price = round(option_price * (1 + slippage), 2)
    option_sym = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)
    print(f"Entry price (with slippage): {entry_price}, Option symbol: {option_sym}")

    confidence = round(np.random.beta(5, 2), 2)
    hour = RNG.randint(10, 15)
    atr = RNG.uniform(2, 6)
    is_swing = RNG.random() < 0.25
    print(f"Confidence: {confidence}, Hour: {hour}, ATR: {atr:.2f}, Is swing trade: {is_swing}")

    indicators = compute_all_indicators(prices, volumes, start_idx)
    print(f"Indicators at start_idx {start_idx}: {indicators}")

    setup_quality = RNG.uniform(0.6, 1.0)
    print(f"Setup quality: {setup_quality:.2f}")

    feature_dict = {
        "confidence": confidence,
        "setup_quality": setup_quality,
        "vix": vix,
        "realized_vol": RNG.uniform(0.1, 0.6),
        "trade_type": int(is_swing),
        "total_signals_today": RNG.randint(1, 7),
        **indicators
    }
    print(f"Feature dict keys: {list(feature_dict.keys())}")

    features_df = build_features_for_trade(feature_dict)
    print(f"Features DF head:\n{features_df.head()}")

    trade_success_prob = float(model_inference.predict_proba(features_df)[0])
    predicted_direction = int(model_inference.predict(features_df)[0])
    class_probabilities = {
        "success": trade_success_prob,
        "failure": 1 - trade_success_prob
    }
    entropy = -sum(p * math.log(p + 1e-9) for p in class_probabilities.values())
    print(f"Trade success prob: {trade_success_prob:.4f}, Predicted direction: {predicted_direction}, Entropy: {entropy:.4f}")

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
    print(f"Base meta state dict: {base_meta_state_dict}")

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
    print(f"Meta entry shape: {meta_entry.shape if meta_entry is not None else None}")
    if meta_entry is None:
        print("Meta entry is None, skipping trade.")
        return None

    action = meta_agent.act(meta_entry)
    print(f"Meta agent action: {action}")

    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    print(f"Trade duration: {duration}")
    if start_idx + duration >= len(prices):
        print("Trade duration exceeds price length, skipping trade.")
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
    print(f"Final price: {final_price}, New option price: {new_option_price}")

    exit_price = round(new_option_price * (1 + slippage), 2)
    gross_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * fill_pct
    total_commission = 2 * COMMISSION_PER_CONTRACT
    trade_result = gross_pnl - total_commission
    print(f"Exit price (with slippage): {exit_price}, Gross PnL: {gross_pnl:.2f}, Trade result after commission: {trade_result:.2f}")

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
    print(f"Meta exit shape: {meta_exit.shape if meta_exit is not None else None}")
    if meta_exit is None:
        print("Meta exit is None, skipping trade.")
        return None

    reward, shaped = reward_shaper.compute_shaped_reward(
        trade_result=trade_result,
        confidence=confidence,
        setup_quality=setup_quality,
        trade_success_prob=trade_success_prob,
        is_swing=is_swing,
        vix=vix,
        predicted_direction=predicted_direction,
        final_price=final_price,
        entry_price=price_sig,
        exit_quality=abs(trade_result) / atr
    )
    print(f"Computed reward: {reward:.4f}, Shaped reward: {shaped:.4f}")

    if shaped < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        print("Shaped reward below threshold and discarded by garbage keep prob.")
        return None

    log_training_example(feature_dict, trade_success_prob, predicted_direction, trade_result)

    return {
        "timestamp": str(datetime.utcnow()),
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
        "pnl": round(trade_result, 2),
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
                logger.debug(f"✅ Trade {trade_idx+1} generated: PnL={log_entry['pnl']}, duration={log_entry['duration']}")
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