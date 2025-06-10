# sim_train_full.py

import os, json, math, time, random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from meta.meta_state   import normalize_meta_state
from meta.meta_agent   import MetaAgent
from meta.reward_shaper import compute_shaped_reward
from utils.telegram_utils import send_telegram_message
from utils.logger       import bot_logger as logger

# ───────── simulation params
SIM_DAYS            = 60
TRADES_PER_DAY      = 10
GBM_MU              = 0.08
GBM_SIGMA           = 0.22
START_PRICE         = 450.0
IV_BASE             = 0.18

META_LOG_PATH       = Path("meta/meta_log.jsonl")
RNG                 = random.Random(42)

meta_agent = MetaAgent()


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
    price_at_signal = prices[start_idx]
    strike      = round(price_at_signal + RNG.uniform(-6, 6), 1)
    option_sym  = make_option_symbol(datetime.utcnow() + timedelta(days=day_idx), strike, option_type)

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

    meta_action = meta_agent.select_action(meta_state)
    meta_param  = meta_agent.interpret_action(meta_action, conf)

    # ───────── realism: delay/slippage/partial
    fill_delay = RNG.randint(1, 5)
    fill_idx   = min(start_idx + fill_delay, len(prices) - 1)
    fill_price = prices[fill_idx]

    # slippage adjustment
    slippage_pct = (fill_price - price_at_signal) / price_at_signal
    slippage_pct += RNG.gauss(0, 0.001)  # ±0.1% extra noise

    # simulate final price movement
    pct_move_underlying = RNG.uniform(-0.6, 0.6)
    raw_pnl_pct = pct_move_underlying * RNG.uniform(80, 120) * 0.3

    # bias toward correct side if confidence is high
    if RNG.random() < confidence:
        raw_pnl_pct = abs(raw_pnl_pct)
    else:
        raw_pnl_pct = -abs(raw_pnl_pct)

    raw_pnl_pct = max(min(raw_pnl_pct, 1.8), -0.9)

    # realism: partial fill
    fill_ratio = round(RNG.uniform(0.6, 1.0), 2)

    trade = {
        "id":     f"SIM.{day_idx}-{step_idx}",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "symbol": "SPY",
        "option_symbol": option_sym,
        "trade_type": 0,
        "confidence": confidence,
        "entry_price": fill_price,
        "slippage_pct": round(slippage_pct * 100, 3),
        "fill_delay_min": fill_delay,
        "fill_ratio": fill_ratio,
        "exit_price":  round(fill_price * (1 + pct_move_underlying / 100), 2),
        "pnl": round(raw_pnl_pct * 100 * fill_ratio, 2),
        "meta_state": meta_state.tolist(),
        "meta_action": int(meta_action),
    }
    return trade


def append_meta_log(trade: dict, vix: float):
    market = { "vix": vix }
    META_LOG_PATH.parent.mkdir(exist_ok=True)
    with META_LOG_PATH.open("a") as fh:
        fh.write(json.dumps({
            "trade":      trade,
            "market":     market,
            "exit_reason": "sim_exit",
            "meta_state":  trade["meta_state"],
            "meta_action": trade["meta_action"],
            "reward":      compute_shaped_reward({
                                "trade": trade,
                                "market": market,
                                "exit_reason": "sim_exit"
                            }),
            "done": True
        }) + "\n")


def simulate():
    logger.info("🧪 Starting synthetic back‑test with realism …")
    current_price = START_PRICE

    for day in range(SIM_DAYS):
        logger.info(f"── Day {day+1}/{SIM_DAYS}")
        minutes_per_day = 390
        prices = gbm_path(minutes_per_day, current_price,
                          GBM_MU/252, GBM_SIGMA/np.sqrt(252), dt=1/390)
        current_price = prices[-1]
        vix_today = RNG.uniform(14, 28)

        for step in range(TRADES_PER_DAY):
            trade = simulate_trade(day, step, prices, vix_today)
            append_meta_log(trade, vix_today)
            time.sleep(0.05)

    logger.info("✅ Simulation finished.")
    send_telegram_message("✅ Simulation finished – launching PPO training …")
    os.system("python3 train_meta_agent.py")


if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")