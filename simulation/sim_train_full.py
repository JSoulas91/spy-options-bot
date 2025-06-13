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
TRADES_PER_DAY      = 10
GBM_MU              = 0.08
GBM_SIGMA           = 0.22
START_PRICE         = 450.0

META_LOG_PATH       = Path("meta/meta_log.jsonl")
RNG                 = random.Random(42)

meta_agent = MetaAgent()

# ╭──────────────────────────────────────────────────────────╮
# │  helpers                                                 │
# ╰──────────────────────────────────────────────────────────╯
def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    prices = [s0]
    for _ in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
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

    move_pct     = RNG.uniform(-0.6, 0.6)
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
        "timestamp":   trade["timestamp"],  # ✅ Now included for reporting
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
        minutes_per_day = 390
        prices = gbm_path(minutes_per_day, current_price,
                          GBM_MU / 252, GBM_SIGMA / np.sqrt(252), dt=1/390)
        current_price = prices[-1]
        vix_today     = RNG.uniform(14, 28)

        for step in range(TRADES_PER_DAY):
            trade = simulate_trade(day, step, prices, vix_today)

            # ✅ Patch to include synthetic OHLCV so ML logger doesn't break
            trade["open"]   = trade["entry_price"]
            trade["high"]   = trade["entry_price"] * 1.01
            trade["low"]    = trade["entry_price"] * 0.99
            trade["close"]  = trade["exit_price"]
            trade["volume"] = int(RNG.uniform(1_000_000, 10_000_000))

            append_meta_log(trade, vix_today)

            # ML logging
            label = 1 if trade["pnl"] > 0 else 0
            try:
                timestamp = datetime.strptime(trade["timestamp"], "%Y-%m-%dT%H:%M:%S")
                close     = trade["entry_price"]
                features  = {
                    "confidence": trade["confidence"],
                    "hour": int(trade["timestamp"][11:13]),
                    "vix": vix_today,
                    "atr": trade["meta_state"][3] if len(trade["meta_state"]) > 3 else 4.0,
                    "pnl": trade["pnl"],
                    "regime_bull": 1 if vix_today < 18 else 0,
                    "regime_bear": 1 if vix_today >= 18 else 0
                }
                log_training_example(timestamp, close, features, label)
            except Exception as e:
                logger.warning(f"Failed to log training example: {e}")

            time.sleep(0.05)

    logger.info("✅ Simulation finished.")
    send_telegram_message("✅ Simulation finished – launching PPO training …")

    try:
        subprocess.run(["python3", "meta/train_meta_agent.py"], check=True)
        logger.info("PPO training completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ PPO training failed: {e}")
        send_telegram_message("⚠️ PPO training failed – check VPS logs.")


if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")