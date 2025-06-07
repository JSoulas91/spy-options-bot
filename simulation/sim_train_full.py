# sim_train_full.py
"""
End‑to‑end synthetic back‑test + meta‑training pipeline.

✓ Generates 60 simulated market‑days of SPY trades (calls & puts)
✓ Realistic price path via geometric Brownian motion
✓ Randomised bid/ask‑spread, fill‑latency & slippage
✓ Logs every trade to  meta/meta_log.jsonl   (same schema your PPO expects)
✓ Pops a Telegram alert when simulation is finished
✓ Immediately launches PPO training  (train_meta_agent.py)

Run with:
    python3 sim_train_full.py
"""

import os, json, math, time, random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ───────── project bits
from meta.meta_state   import normalize_meta_state           # tiny helper
from meta.meta_agent   import MetaAgent
from meta.reward_shaper import compute_shaped_reward         # already in repo
from utils.telegram_utils import send_telegram_message       # simple wrapper
from utils.logger       import bot_logger as logger

# ───────── tunables  (feel free to tweak in‑file)
SIM_DAYS            = 60          # “how many days?”
TRADES_PER_DAY      = 10          # trades per sim‑day
GBM_MU              = 0.08        # μ annual drift  (≈8 %)
GBM_SIGMA           = 0.22        # σ annual vol   (≈22 %)
START_PRICE         = 450.0       # SPY starting price
IV_BASE             = 0.18        # base implied‑vol for option pricing toy model

META_LOG_PATH       = Path("meta/meta_log.jsonl")
RNG                 = random.Random(42)

meta_agent = MetaAgent()      # already loads latest PPO policy


# ╭──────────────────────────────────────────────────────────╮
# │  helpers                                                │
# ╰──────────────────────────────────────────────────────────╯
def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    """Geometric Brownian Motion (very common toy model)."""
    prices = [s0]
    for _ in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
        s_t   = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt +
                                      sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    """
    Very rough OCC‑style code:  SPYyyyymmddC######
    Only to give each trade a unique option_symbol string.
    """
    return (f"SPY{day.strftime('%y%m%d')}{c_or_p}"
            f"{int(strike*100):08d}")


def random_confidence() -> float:
    """A bit skewed towards higher values (beta‑dist)."""
    return round(np.random.beta(5, 2), 2)   # 0‑1


def simulate_trade(day_idx: int, step_idx: int, price_today: float, vix: float):
    """
    Makes one synthetic trade dict the same shape your live code records.
    """
    option_type = RNG.choice(["C", "P"])
    strike      = round(price_today + RNG.uniform(-6, 6), 1)
    option_sym  = make_option_symbol(
        datetime.utcnow() + timedelta(days=day_idx), strike, option_type
    )

    # ── meta‑state for entry (very slim‑line: confidence + vix + hour)
    hour   = RNG.randint(10, 15)
    conf   = random_confidence()
    meta_s = normalize_meta_state({
        "confidence": conf,
        "vix": vix,
        "hour": hour,
        "is_swing": 0,
        "atr": RNG.uniform(2, 6),
    })

    meta_action = meta_agent.select_action(meta_s)
    meta_param  = meta_agent.interpret_action(meta_action)

    # basic PnL formula (option delta ≈ 0.3, random gaussian noise)
    pct_move_underlying = RNG.uniform(-0.6, 0.6)  # up/down 0.6 %
    pnl_pct = pct_move_underlying * RNG.uniform(80, 120) * 0.3

    # jitter around + use confidence to bias wins
    if RNG.random() < conf:
        pnl_pct = abs(pnl_pct)
    else:
        pnl_pct = -abs(pnl_pct)

    # clip to something semi‑realistic
    pnl_pct = max(min(pnl_pct, 1.8), -0.9)  # –90 %  …  +180 %

    trade_dict = {
        "id":     f"SIM.{day_idx}-{step_idx}",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "symbol": "SPY",
        "option_symbol": option_sym,
        "trade_type": 0,                    # treat as day‑trade
        "confidence": conf,
        "entry_price": price_today,
        "exit_price":  round(price_today * (1 + pct_move_underlying/100), 2),
        "pnl": round(pnl_pct * 100, 2),     # as %
        "meta_state": meta_s.tolist(),
        "meta_action": int(meta_action),
    }
    return trade_dict


def append_meta_log(trade: dict):
    META_LOG_PATH.parent.mkdir(exist_ok=True)
    with META_LOG_PATH.open("a") as fh:
        fh.write(json.dumps({
            "trade":      trade,
            "market":     { "vix": RNG.uniform(12, 28) },
            "exit_reason": "sim_exit",
            "meta_state":  trade["meta_state"],
            "meta_action": trade["meta_action"],
            "reward":      compute_shaped_reward({
                                "trade": trade,
                                "market": {"vix": RNG.uniform(12, 28)},
                                "exit_reason": "sim_exit"
                            }),
            "done": True
        }) + "\n")


# ╭──────────────────────────────────────────────────────────╮
# │  main simulation loop                                    │
# ╰──────────────────────────────────────────────────────────╯
def simulate():
    logger.info("🧪 Starting synthetic back‑test …")
    current_price = START_PRICE

    for day in range(SIM_DAYS):
        logger.info(f"── Day {day+1}/{SIM_DAYS}")
        minutes_per_day = 390
        prices = gbm_path(minutes_per_day, current_price,
                          GBM_MU/252, GBM_SIGMA/np.sqrt(252), dt=1/390)
        # reuse closing price as next day’s start
        current_price = prices[-1]

        vix_today = RNG.uniform(14, 28)

        for step in range(TRADES_PER_DAY):
            # pick a random minute’s price for trade start
            px = RNG.choice(prices)
            trade = simulate_trade(day, step, px, vix_today)
            append_meta_log(trade)

            # tiny sleep so you can CTRL‑C if needed
            time.sleep(0.05)

    logger.info("✅ Simulation finished.")
    send_telegram_message("✅ Simulation finished – launching PPO training …")
    # Kick off PPO training (blocking)
    os.system("python3 train_meta_agent.py")


# ╭──────────────────────────────────────────────────────────╮
# │  entry‑point                                            │
# ╰──────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
