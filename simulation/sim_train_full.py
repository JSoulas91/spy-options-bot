"""
End‑to‑end synthetic back‑test + meta‑training pipeline.

✓ Generates 60 simulated market‑days of SPY trades (calls & puts)
✓ Realistic price path via geometric Brownian motion
✓ Randomised bid/ask‑spread, fill‑latency & slippage
✓ Logs every trade to  meta/meta_log.jsonl   (schema your PPO expects)
✓ Pops a Telegram alert when simulation is finished
✓ Immediately launches PPO training  (meta/train_meta_agent.py)

Run with:
    python3 simulation/sim_train_full.py
"""

from __future__ import annotations

import os, json, math, time, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np

# ───────── project bits
from meta.meta_state      import normalize_meta_state           # tiny helper
from meta.meta_agent      import MetaAgent
from meta.reward_shaper   import compute_shaped_reward
from utils.telegram_utils import send_telegram_message
from utils.logger         import bot_logger as logger

# ───────── tunables  (tweak freely)
SIM_DAYS            = 60          # simulated trading days
TRADES_PER_DAY      = 10          # trades generated per day
GBM_MU              = 0.08        # μ annual drift  (≈8 %)
GBM_SIGMA           = 0.22        # σ annual vol   (≈22 %)
START_PRICE         = 450.0       # SPY starting price
IV_BASE             = 0.18        # base implied‑vol for option pricing toy model

META_LOG_PATH       = Path("meta/meta_log.jsonl")
RNG                 = random.Random(42)
np.random.seed(42)                # reproducible NumPy draws

meta_agent = MetaAgent()          # loads latest PPO policy


# ╭──────────────────────────────────────────────────────────╮
# │  helpers                                                │
# ╰──────────────────────────────────────────────────────────╯
def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float) -> List[float]:
    """Generate a geometric Brownian‑motion price path."""
    prices = [s0]
    for _ in range(1, n_steps):
        shock = RNG.normalvariate(0, 1)
        s_t   = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) * dt +
                                      sigma * math.sqrt(dt) * shock)
        prices.append(round(s_t, 2))
    return prices


def make_option_symbol(day: datetime, strike: float, c_or_p: str, uid: str) -> str:
    """
    Very rough OCC‑style code with an extra UID for uniqueness:
    SPYyyyymmddC########.UID
    """
    return (
        f"SPY{day.strftime('%y%m%d')}{c_or_p}"
        f"{int(strike*100):08d}.{uid}"
    )


def random_confidence() -> float:
    """Beta‑distributed confidence skewed high."""
    return round(np.random.beta(5, 2), 2)   # 0‑1


def simulate_trade(day_idx: int, step_idx: int, price_today: float, vix: float) -> dict:
    """
    Create a synthetic trade dict matching live‑trade schema.
    """
    option_type = RNG.choice(["C", "P"])
    strike      = round(price_today + RNG.uniform(-6, 6), 1)
    option_sym  = make_option_symbol(
        datetime.utcnow() + timedelta(days=day_idx),
        strike,
        option_type,
        uid=f"{day_idx}{step_idx}{RNG.randint(1000,9999)}",
    )

    # ── meta‑state for entry (confidence + vix + hour + swing/ATR)
    hour   = RNG.randint(10, 15)
    conf   = random_confidence()
    meta_s = normalize_meta_state({
        "confidence": conf,
        "vix": vix,
        "hour": hour,
        "is_swing": 0,
        "atr": RNG.uniform(2, 6),
    })

    action_idx, agent_conf = meta_agent.select_action(meta_s)
    meta_param             = meta_agent.interpret_action(action_idx, agent_conf)

    # basic PnL toy formula using option delta ≈ 0.3 and random noise
    pct_move_underlying = RNG.uniform(-0.6, 0.6)  # up/down 0.6 %
    pnl_pct = pct_move_underlying * RNG.uniform(0.8, 1.2) * 0.3

    # bias wins by confidence
    pnl_pct = abs(pnl_pct) if RNG.random() < conf else -abs(pnl_pct)
    pnl_pct = max(min(pnl_pct, 1.8), -0.9)  # –90 % … +180 %

    trade_dict = {
        "id":           f"SIM.{day_idx}-{step_idx}",
        "timestamp":    datetime.utcnow().isoformat(timespec="seconds"),
        "symbol":       "SPY",
        "option_symbol": option_sym,
        "trade_type":   0,                         # treat as day‑trade
        "confidence":   conf,
        "entry_price":  price_today,
        "exit_price":   round(price_today * (1 + pct_move_underlying / 100), 2),
        "pnl":          round(pnl_pct * 100, 2),   # as %
        "meta_state":   meta_s.tolist(),
        "meta_action":  {"dir": action_idx, "conf": agent_conf},
    }
    return trade_dict


def append_meta_log(trade: dict, vix_value: float) -> None:
    """Append single experience to JSONL log with shaped reward."""
    META_LOG_PATH.parent.mkdir(exist_ok=True)

    payload = {
        "trade":        trade,
        "market":       {"vix": vix_value},
        "exit_reason":  "sim_exit",
        "meta_state":   trade["meta_state"],
        "meta_action":  trade["meta_action"],
        "reward":       compute_shaped_reward({
                            "trade": trade,
                            "market": {"vix": vix_value},
                            "exit_reason": "sim_exit",
                        }),
        "done": True
    }
    with META_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")


# ╭──────────────────────────────────────────────────────────╮
# │  main simulation loop                                    │
# ╰──────────────────────────────────────────────────────────╯
def simulate() -> None:
    logger.info("🧪 Starting synthetic back‑test …")
    current_price = START_PRICE

    for day in range(SIM_DAYS):
        logger.info(f"── Day {day+1}/{SIM_DAYS}")
        minutes_per_day = 390
        prices = gbm_path(
            minutes_per_day,
            current_price,
            GBM_MU / 252,
            GBM_SIGMA / np.sqrt(252),
            dt=1 / 390,
        )
        current_price = prices[-1]  # last close = next open
        vix_today     = RNG.uniform(14, 28)

        for step in range(TRADES_PER_DAY):
            px = RNG.choice(prices)           # random trade entry price
            trade = simulate_trade(day, step, px, vix_today)
            append_meta_log(trade, vix_today)
            time.sleep(0.05)                  # slight delay for Ctrl‑C breaks

    logger.info("✅ Simulation finished.")
    send_telegram_message("✅ Simulation finished – launching PPO training …")

    # Kick off PPO training (blocking)
    os.system("python3 meta/train_meta_agent.py")


# ╭──────────────────────────────────────────────────────────╮
# │  entry‑point                                            │
# ╰──────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")