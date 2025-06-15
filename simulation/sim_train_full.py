import os, json, math, time, random, subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from meta.meta_state    import normalize_meta_state
from meta.meta_agent    import MetaAgent
from meta.reward_shaper import compute_shaped_reward
from utils.telegram_utils import send_telegram_message
from utils.logger        import bot_logger as logger
from ml.logger           import log_training_example  # ML logger import

# ───────── simulation params
SIM_DAYS            = 60
START_PRICE         = 450.0
GBM_MU              = 0.08

META_LOG_PATH       = Path("meta/meta_log.jsonl")
RNG                 = random.Random(42)

meta_agent = MetaAgent()

# ╭──────────────────────────────────────────────────────────╮
# │  helpers                                                 │
# ╰──────────────────────────────────────────────────────────╯
def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    prices = [s0]
    for i in range(1, n_steps):
        intraday_vol_boost = 1.2 if i < 60 or i > 330 else 1.0
        shock = RNG.normalvariate(0, 1) * intraday_vol_boost
        s_t   = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt +
                                      sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    return f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"


def random_confidence() -> float:
    return round(np.random.beta(5, 2), 2)


def simulate_trade(day_idx: int, step_idx: int, prices: list[float], vix: float):
    option_type = RNG.choice(["C", "P"])
    start_idx   = RNG.randint(30, len(prices) - 30)
    price_sig   = prices[start_idx]
    strike      = round(price_sig + RNG.uniform(-6, 6), 1)
    option_sym  = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx),
                                     strike, option_type)

    hour        = RNG.randint(10, 15)
    confidence  = random_confidence()
    atr         = RNG.uniform(2, 6)
    meta_state  = normalize_meta_state({
        "confidence": confidence,
        "vix": vix,
        "hour": hour,
        "is_swing": 0,
        "atr": atr,
    })

    action_idx, agent_conf = meta_agent.select_action(meta_state)
    meta_agent.interpret_action(action_idx, agent_conf)

    fill_delay  = RNG.randint(1, 5)
    fill_idx    = min(start_idx + fill_delay, len(prices) - 1)
    fill_price  = prices[fill_idx]

    slippage_pct = (fill_price - price_sig) / price_sig + RNG.gauss(0, 0.001)

    # Flash spike/crash logic (~2% chance)
    if RNG.random() < 0.02:
        move_pct = RNG.uniform(-3.0, 3.0)  # Extreme scenario
    else:
        move_pct = RNG.uniform(-0.6, 0.6)

    raw_pnl_pct  = move_pct * RNG.uniform(0.8, 1.2) * 0.3
    raw_pnl_pct  = abs(raw_pnl_pct) if RNG.random() < confidence else -abs(raw_pnl_pct)
    raw_pnl_pct  = max(min(raw_pnl_pct, 1.8), -0.9)

    fill_ratio   = round(RNG.uniform(0.6, 1.0), 2)

    trade = {
        "id":            f"SIM.{day_idx}-{step_idx}",
        "timestamp":     datetime.utcnow().isoformat(timespec="seconds"),
        "symbol":        "SPY",
        "option_symbol": option_sym,
        "trade_type":    0,
        "confidence":    confidence,
        "entry_price":   fill_price,
        "slippage_pct":  round(slippage_pct * 100, 3),
        "fill_delay_min": fill_delay,
        "fill_ratio":    fill_ratio,
        "exit_price":    round(fill_price * (1 + move_pct / 100), 2),
        "pnl":           round(raw_pnl_pct * 100 * fill_ratio, 2),
        "meta_state":    meta_state.tolist(),
        "meta_action":   int(action_idx),
    }
    return trade


def append_meta_log(trade: dict, vix_val: float):
    META_LOG_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "timestamp":   trade["timestamp"],
        "trade":       trade,
        "market":      {"vix": vix_val},
        "exit_reason": "sim_exit",
        "meta_state":  trade["meta_state"],
        "meta_action": trade["meta_action"],
        "reward":      compute_shaped_reward({
                           "trade": trade,
                           "market": {"vix": vix_val},
                           "exit_reason": "sim_exit"
                       }),
        "done": True
    }
    with META_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


# ╭──────────────────────────────────────────────────────────╮
# │  main simulation loop                                    │
# ╰──────────────────────────────────────────────────────────╯
def simulate():
    logger.info("🧪 Starting synthetic back‑test with realism …")
    current_price = START_PRICE

    for day in range(SIM_DAYS):
        logger.info(f"── Day {day+1}/{SIM_DAYS}")

        # Regime-driven volatility
        vix_today = RNG.uniform(14, 28)
        sigma     = 0.15 + (vix_today - 12) * 0.01

        minutes_per_day = 390
        prices = gbm_path(minutes_per_day, current_price,
                          GBM_MU / 252, sigma, dt=1/390)
        current_price = prices[-1]

        # Vary trades per day by volatility
        if vix_today < 16:
            trades_today = RNG.randint(8, 12)
        elif vix_today < 22:
            trades_today = RNG.randint(6, 9)
        else:
            trades_today = RNG.randint(3, 6)

        for step in range(trades_today):
            trade = simulate_trade(day, step, prices, vix_today)
            append_meta_log(trade, vix_today)

            # ML logging
            label = 1 if trade["pnl"] > 0 else 0
            try:
                timestamp = datetime.strptime(trade["timestamp"], "%Y-%m-%dT%H:%M:%S")
                close     = trade["entry_price"]

                open_price = round(close * RNG.uniform(0.995, 1.005), 2)
                high_price = round(max(open_price, close) * RNG.uniform(1.0, 1.01), 2)
                low_price  = round(min(open_price, close) * RNG.uniform(0.99, 1.0), 2)
                volume     = int(RNG.uniform(1000, 10000))

                features  = {
                    "confidence": trade["confidence"],
                    "hour": int(trade["timestamp"][11:13]),
                    "vix": vix_today,
                    "atr": trade["meta_state"][3] if len(trade["meta_state"]) > 3 else 4.0,
                    "pnl": trade["pnl"],
                    "regime_bull": 1 if vix_today < 18 else 0,
                    "regime_bear": 1 if vix_today >= 18 else 0,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close,
                    "volume": volume
                }
                log_training_example(timestamp, close, features, label)
            except Exception as e:
                logger.warning(f"Failed to log training example: {e}")

            time.sleep(0.05)

    logger.info("✅ Simulation finished.")
    send_telegram_message("✅ Simulation finished.")  # Training disabled


if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")