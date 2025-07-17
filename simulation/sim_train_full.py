import os
import json
import math
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import traceback

from dotenv import load_dotenv
load_dotenv()

from typing import Tuple, List, Dict, Optional
from config import ENABLE_DYNAMIC_SIZING, MIN_POSITION_SIZE, MAX_POSITION_SIZE, DEFAULT_POSITION_SIZE
from meta.meta_agent import MetaAgent
from meta.reward_shaper import RewardShaper
from utils.telegram_utils import send_telegram_message
from utils.logger import bot_logger as logger
from ml.logger import log_training_example
from ml.model_inference import ModelInference
from ml.feature_pipeline import build_features_for_trade

reward_shaper = RewardShaper()

# ───────── simulation params ───────────────
SIM_DAYS = 1200
TRADES_PER_DAY = 12
GBM_MU = 0.08
GBM_SIGMA = 0.22
START_PRICE = 450.0
WARM_UP_DAYS = 110
PAD_VAL = 0.5
STATE_SEQUENCE_LENGTH = 20  # Adjust if your model uses more or fewer timesteps
STATE_DIM = 83             # Must match what your model expects per timestep
_DYNAMIC: Dict[str, Tuple[Tuple[float, float], float]] = {}
_DYN_TTL = 3600
ACCUMULATED_CLOSES = []
ACCUMULATED_VOLUMES = []
TRADE_HISTORY = []

long_term_data = {
    "5d": pd.DataFrame(),
    "10d": pd.DataFrame(),
    "15d": pd.DataFrame(),
    "1mo": pd.DataFrame(),
    "3mo": pd.DataFrame(),
    "6mo": pd.DataFrame(),
}

DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "RSI": (0, 100),
    "MACD": (-5, 5),
    "EMA_DIST": (-10, 10),
    "VOL": (0, 10_000_000),
    "CONF": (0, 1),
    "DURATION": (0, 390),
    "PROFIT": (-1, 1),
    "VIX": (10, 40),
    "SPY_ABS": (350, 500),
    "IV": (0, 1),
    "DELTA": (-1, 1),
    "SIZE": (0, MAX_POSITION_SIZE),
}

META_LOG_PATH = Path("meta/meta_log.jsonl")
RNG = random.Random(42)
GARBAGE_KEEP_PROB = 0.05
COMMISSION_PER_CONTRACT = 0.35
CONTRACT_MULTIPLIER = 100  # Options multiplier

meta_agent = MetaAgent()
model_inference = ModelInference()

print(f"SIM_DAYS={SIM_DAYS}, TRADES_PER_DAY={TRADES_PER_DAY}, START_PRICE={START_PRICE}")

def summarize_simulation_results(trade_history: list, skipped_count: int, diagnostics: dict = None):
    from collections import defaultdict

    pnl_buckets = defaultdict(int)
    total_pnl = 0.0
    rewards = []
    wins = 0
    losses = 0

    for trade in trade_history:
        pct = trade.get("pct_pnl", 0)
        reward = trade.get("reward", 0)
        rewards.append(reward)
        total_pnl += pct
        if pct >= 0:
            wins += 1
        else:
            losses += 1

        abs_pct = abs(pct)
        bucket = f"{int(abs_pct // 10) * 10}-{int(abs_pct // 10) * 10 + 10}%"
        key = f"+{bucket}" if pct >= 0 else f"-{bucket}"
        if abs_pct >= 50:
            key = "+50%+" if pct >= 0 else "-50%+"
        pnl_buckets[key] += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    std_reward = np.std(rewards) if rewards else 1.0
    sharpe = avg_reward / std_reward if std_reward != 0 else 0.0

    msg_lines = [
        "📊 *Simulation Summary*",
        f"Total trades: *{len(trade_history)}*",
        f"✅ Wins: *{wins}* | ❌ Losses: *{losses}*",
        f"📈 Total %PnL: *{total_pnl:.2f}%*",
        f"🎯 Average reward: *{avg_reward:.4f}*",
        f"⚖️ Sharpe ratio: *{sharpe:.2f}*",
        f"🚫 Trades skipped: *{skipped_count}*",
        "",
        "📊 *PnL Buckets*"
    ]

    for bucket in sorted(pnl_buckets.keys(), key=lambda x: (x[0], int(x.strip("+-%").split('-')[0]))):
        msg_lines.append(f"{bucket}: *{pnl_buckets[bucket]}*")

    if diagnostics:
        msg_lines.append("")
        msg_lines.append("🧠 *Meta Agent Diagnostics*")
        for k, v in diagnostics.items():
            msg_lines.append(f"{k}: *{v}*")
            
    return "\n".join(msg_lines)

def debug_inputs(label: str, **kwargs):
    logger.debug(f"\n[DEBUG] {label} input diagnostics:")
    for name, val in kwargs.items():
        if isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)):
            logger.debug(f"  - {name}: type={type(val)}, shape={val.shape}")
        elif isinstance(val, list):
            logger.debug(f"  - {name}: type=list, length={len(val)}")
        elif isinstance(val, dict):
            logger.debug(f"  - {name}: type=dict, keys={list(val.keys())}")
        else:
            logger.debug(f"  - {name}: type={type(val)}, value={val}")
    logger.debug("-" * 60)

def halt_on_error(context: str, err: Exception, **inputs):
    logger.error(f"\n[ERROR] Failure in: {context}")
    logger.error("Inputs at failure:")
    for k, v in inputs.items():
        logger.error(f"  - {k}: {type(v)}, {str(v)[:300]}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())
    raise err

def generate_dummy_trade():
    return {
        "pct_pnl": RNG.uniform(-0.05, 0.05),  # +/- 5% PnL
        "duration": RNG.randint(10, 120),     # 10 to 120 minutes
        "position_size": RNG.uniform(0.5, 2.0),
        "classifier_output": [RNG.uniform(0, 1) for _ in range(3)],
        "classifier_features": {f"feat_{i}": RNG.uniform(-1, 1) for i in range(29)},  # ✅ dict, not list
        "meta_entry": [RNG.uniform(0, 1) for _ in range(64)],
        "meta_exit": [RNG.uniform(0, 1) for _ in range(64)],
        "option_data": {
            "iv": RNG.uniform(0.1, 0.6),
            "delta": RNG.uniform(0.2, 0.8),
        },
        "indicators": {
            "rsi_14": RNG.uniform(30, 70),
            "macd": RNG.uniform(-1, 1),
            "macd_hist": RNG.uniform(-1, 1),
            "ema_20": RNG.uniform(400, 500),
            "price": RNG.uniform(400, 500),
            "atr_14": RNG.uniform(1, 10),
            "adx_14": RNG.uniform(10, 30),
            "vwap": RNG.uniform(400, 500),
            "bb_upper": RNG.uniform(410, 520),
            "bb_lower": RNG.uniform(380, 490),
            "vix": RNG.uniform(14, 28),
        }
    }


def is_padded(meta):
    try:
        meta = np.array(meta)
        return np.allclose(meta, 0.5, atol=1e-6)
    except Exception as e:
        halt_on_error("is_padded", e, meta=meta)
        
import os
import json
import numpy as np

def write_to_meta_log(trade: dict, path: str = "meta/meta_log.jsonl", error_path: str = "meta/meta_log_errors.jsonl"):
    """
    Writes a full trade record to meta_log.jsonl for meta-agent training or inspection.
    Adds deep debugging, including detection of padded or malformed meta states.
    """
    try:
        entry_state = trade.get("entry_state")
        exit_state = trade.get("exit_state")

        def shape_or_type(x):
            return np.shape(x) if hasattr(x, "__len__") else type(x)

        logger.debug(f"📝 Writing trade_id={trade.get('trade_idx')}, PnL={trade.get('pct_pnl')}, Reward={trade.get('shaped_reward')}")
        logger.debug(f"📐 entry_state: {shape_or_type(entry_state)}, exit_state: {shape_or_type(exit_state)}")

        # Extra checks for padded or broken states
        padded_entry = isinstance(entry_state, (list, np.ndarray)) and all(v == 0.5 for v in entry_state)
        padded_exit = isinstance(exit_state, (list, np.ndarray)) and all(v == 0.5 for v in exit_state)
        if padded_entry or padded_exit:
            logger.warning(f"⚠️ Padded Meta-State detected | trade_id={trade.get('trade_idx')} | padded_entry={padded_entry}, padded_exit={padded_exit}")
            # Optionally dump bad trades to a separate error log for inspection
            try:
                with open(error_path, "a") as ef:
                    ef.write(json.dumps(trade, default=str) + "\n")
                logger.debug(f"🗃️ Dumped bad trade to {error_path}")
            except Exception as e:
                logger.error(f"❌ Failed to write bad meta state to {error_path}: {e}")

        # Now write to main meta log
        with open(path, "a") as f:
            f.write(json.dumps({
                "timestamp": trade.get("timestamp"),
                "day": trade.get("day"),
                "trade_idx": trade.get("trade_idx"),
                "pct_pnl": trade.get("pct_pnl"),
                "shaped_reward": trade.get("shaped_reward"),
                "meta_entry_state": entry_state,
                "meta_exit_state": exit_state,
                "meta_action": trade.get("meta_action"),
                "agent_confidence": trade.get("agent_confidence"),
                "classifier": trade.get("classifier"),
                "features": trade.get("features"),
                "option": trade.get("option"),
                "duration": trade.get("duration"),
                "fill_pct": trade.get("fill_pct"),
                "position_size": trade.get("position_size"),
                "strike": trade.get("strike"),
                "type": trade.get("type"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "final_price": trade.get("final_price"),
            }, default=str) + "\n")

    except Exception as e:
        logger.error(f"❌ Failed to write to meta_log.jsonl: {e}", exc_info=True)
        
def gbm_path(n_steps: int, s0: float, mu: float, sigma: float, dt: float):
    try:
        prices = [s0]
        for _ in range(1, n_steps):
            shock = RNG.normalvariate(0, 1)
            s_t = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock)
            prices.append(round(s_t, 2))
        logger.debug(f"✅ GBM path generated with {n_steps} steps from {s0} starting price.")
        logger.debug(f"First 5 prices: {prices[:5]}")
        logger.debug(f"Last 5 prices: {prices[-5:]}")
        return prices
    except Exception as e:
        halt_on_error("gbm_path", e, n_steps=n_steps, s0=s0, mu=mu, sigma=sigma, dt=dt)

def normalize(value, value_range):
    try:
        min_val, max_val = value_range
        if max_val == min_val:
            return PAD_VAL
        return max(PAD_VAL, min(1.0, (value - min_val) / (max_val - min_val)))
    except Exception as e:
        halt_on_error("normalize", e, value=value, value_range=value_range)

def _calc_range(feat: str, long_term: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    vals: List[float] = []

    def to_list_safe(x):
        try:
            if hasattr(x, "tolist"):
                return x.tolist()
            elif isinstance(x, list):
                return x
            else:
                return list(x)
        except Exception:
            logger.warning(f"[WARN] Failed to convert feature column to list for {feat}")
            return []

    for idx, df in enumerate(long_term.values()):
        if df is None or not isinstance(df, pd.DataFrame):
            logger.warning(f"⚠️ long_term[{idx}] is not a valid DataFrame: type={type(df)}")
            continue
        if df.empty:
            continue

        if feat == "EMA_DIST":
            if "price" in df.columns and "ema_20" in df.columns:
                diff = df["price"] - df["ema_20"]
                vals.extend(to_list_safe(diff))
            else:
                logger.warning(f"⚠️ EMA_DIST missing 'price' or 'ema_20' in long_term[{idx}] → cols: {df.columns}")
        else:
            if feat in df.columns:
                vals.extend(to_list_safe(df[feat]))
            else:
                logger.warning(f"⚠️ Feature '{feat}' missing in long_term[{idx}] → cols: {df.columns}")

    return (min(vals), max(vals)) if vals else DEFAULT_RANGES[feat]

def get_range(feat: str, long_term: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    try:
        if not isinstance(long_term, dict):
            raise TypeError(f"💥 get_range expected dict, got {type(long_term)}")
        now = time.time()
        if feat in _DYNAMIC and now - _DYNAMIC[feat][1] < _DYN_TTL:
            return _DYNAMIC[feat][0]
        rng = _calc_range(feat, long_term)
        if rng[0] == rng[1]:
            rng = DEFAULT_RANGES[feat]
        _DYNAMIC[feat] = (rng, now)
        return rng
    except Exception as e:
        halt_on_error("get_range", e, feat=feat, long_term=long_term)

def update_long_term_stats(long_term_data: dict, features: dict):
    """
    Updates each rolling time window (e.g., '5d', '10d') in long_term_data with a new row
    of classifier features. Ensures each buffer remains a proper DataFrame with shape (N, 83).
    """
    try:
        logger.debug(f"🔁 Updating long_term_data with {len(features)} features")

        # Step 1: Clean and validate scalar values
        cleaned = {}
        for k, v in features.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 1:
                    cleaned[k] = v[0]
                else:
                    logger.error(f"🚫 Feature '{k}' is non-scalar: {v} (type={type(v)})")
                    raise ValueError(f"Feature '{k}' must be scalar, got {v}")
            else:
                cleaned[k] = v

        # Step 2: Construct new one-row DataFrame
        row_df = pd.DataFrame([cleaned])
        logger.debug(f"✅ Constructed row_df for update: shape={row_df.shape}")

        # Step 3: Update each time window key
        for key in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            if key not in long_term_data:
                logger.warning(f"⚠️ Creating new long_term_data[{key}] buffer")
                long_term_data[key] = row_df.copy()
            else:
                if not isinstance(long_term_data[key], pd.DataFrame):
                    logger.error(f"❌ long_term_data[{key}] is not a DataFrame (got {type(long_term_data[key])})")
                    raise TypeError(f"long_term_data[{key}] must be a DataFrame")

                long_term_data[key] = pd.concat([long_term_data[key], row_df], ignore_index=True)

            # Step 4: Truncate to avoid memory bloat
            if len(long_term_data[key]) > 500:
                long_term_data[key] = long_term_data[key].iloc[-500:]

            logger.debug(f"📈 Updated long_term_data[{key}]: shape={long_term_data[key].shape}")

    except Exception as e:
        halt_on_error("update_long_term_stats", e, long_term_data=long_term_data, features=features)
            
def _pad(state: List[float]) -> np.ndarray:
    try:
        state = np.array(state, dtype=np.float32)
        padded = np.full(STATE_DIM, PAD_VAL, dtype=np.float32)
        length = min(len(state), STATE_DIM)
        padded[:length] = state[:length]
        return padded
    except Exception as e:
        halt_on_error("_pad", e, state=state)

def _regime_one_hot(regime: str) -> List[float]:
    try:
        mapping = {"bull": [1.0, 0.0, 0.0], "bear": [0.0, 1.0, 0.0], "sideways": [0.0, 0.0, 1.0]}
        result = mapping.get(regime.lower(), [PAD_VAL, PAD_VAL, PAD_VAL])
        if result == [PAD_VAL, PAD_VAL, PAD_VAL]:
            logger.warning(f"[WARN] Unknown regime '{regime}', defaulting to PAD_VALs.")
        return result
    except Exception as e:
        halt_on_error("_regime_one_hot", e, regime=regime)

def _classify_regime(day_bar: dict, vix_val: float) -> str:
    try:
        if vix_val > 30 or day_bar.get("rsi", 50) < 40:
            return "bear"
        elif vix_val < 20 and day_bar.get("rsi", 50) > 60:
            return "bull"
        else:
            return "sideways"
    except Exception as e:
        halt_on_error("_classify_regime", e, day_bar=day_bar, vix_val=vix_val)

def fetch_vix_price() -> Optional[float]:
    try:
        # Placeholder — replace with live API call or historical lookup
        return 18.0
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch VIX price: {str(e)}")
        return None

def get_minutes_since_open() -> int:
    try:
        from datetime import datetime
        now = datetime.utcnow()
        market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 AM ET
        delta = now - market_open
        minutes = max(0, delta.seconds // 60)
        return minutes
    except Exception as e:
        halt_on_error("get_minutes_since_open", e)

def summarise_past(past_trades: List[dict], profit_range: tuple, dur_range: tuple, size_range: tuple) -> List[float]:
    try:
        if len(past_trades) < 3:
            raise ValueError("Not enough past trades to summarize without padding")

        # Only take last 3 trades
        recent = past_trades[-3:]

        features = []
        for trade in recent:
            pct_pnl = normalize(trade.get("pct_pnl", 0.0), profit_range)
            duration = normalize(trade.get("duration", 0.0), dur_range)
            size = normalize(trade.get("position_size", 1.0), size_range)

            # Start with core trade info
            row = [pct_pnl, duration, size]

            clf_feats = trade.get("classifier_features", {})
            clf_values = [float(clf_feats[k]) for k in sorted(clf_feats.keys()) if isinstance(clf_feats[k], (int, float))]

            row.extend(clf_values)

            features.extend(row)

        # Trim to exactly 29
        return features[:29]

    except Exception as e:
        halt_on_error("summarise_past", e, past_trades=past_trades)

def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    try:
        if c_or_p not in ("C", "P"):
            raise ValueError(f"Invalid option type '{c_or_p}', must be 'C' or 'P'")
        symbol = f"SPY{day.strftime('%y%m%d')}{c_or_p.upper()}{int(strike * 100):08d}"
        logger.debug(f"✅ Option symbol created: {symbol}")
        return symbol
    except Exception as e:
        halt_on_error("make_option_symbol", e, day=day, strike=strike, c_or_p=c_or_p)

def construct_bars(prices, volumes, interval, start_time=None):
    try:
        if start_time is None:
            start_time = datetime(2025, 1, 1, 9, 30)

        # Type checks
        if not isinstance(prices, (list, np.ndarray)) or not isinstance(volumes, (list, np.ndarray)):
            raise TypeError(f"Prices and volumes must be list or np.ndarray. Got types: prices={type(prices)}, volumes={type(volumes)}")

        prices = list(prices)
        volumes = list(volumes)

        if len(prices) != len(volumes):
            logger.warning(f"⚠️ construct_bars: length mismatch - prices={len(prices)}, volumes={len(volumes)}")
            return pd.DataFrame()

        if len(prices) < interval:
            logger.warning(f"⚠️ construct_bars: not enough data to form one bar - len(prices)={len(prices)}, interval={interval}")
            return pd.DataFrame()

        bars = []
        for i in range(0, len(prices) - interval + 1, interval):
            chunk = prices[i:i + interval]
            vol_chunk = volumes[i:i + interval]

            if len(chunk) < interval or len(vol_chunk) < interval:
                logger.debug(f"⚠️ construct_bars: incomplete chunk at i={i} - chunk_len={len(chunk)}, vol_len={len(vol_chunk)}")
                continue

            bar_time = start_time + timedelta(minutes=i)

            bar = {
                "timestamp": bar_time,
                "open": float(chunk[0]),
                "high": float(max(chunk)),
                "low": float(min(chunk)),
                "close": float(chunk[-1]),
                "volume": float(sum(vol_chunk))
            }

            bars.append(bar)

        df = pd.DataFrame(bars)
        logger.debug(f"✅ construct_bars: constructed {len(df)} bars at interval={interval}min from {start_time.strftime('%Y-%m-%d %H:%M')}")
        return df

    except Exception as e:
        halt_on_error("construct_bars", e, prices_len=len(prices), volumes_len=len(volumes), interval=interval, start_time=start_time)
        return pd.DataFrame()

def compute_all_indicators(prices, volumes, idx):
    try:
        indicators = {}
        start_idx = max(0, idx - 100)
        window = prices[start_idx:idx + 1]
        vol_window = volumes[start_idx:idx + 1]

        logger.debug(f"[TRACE] compute_all_indicators: window len={len(window)}, vol_window len={len(vol_window)}")

        # Ensure numeric
        closes_series = pd.to_numeric(pd.Series(window), errors="coerce")
        vol_series = pd.to_numeric(pd.Series(vol_window), errors="coerce").fillna(0.0)

        valid_mask = closes_series.notna()
        if valid_mask.sum() == 0:
            logger.error(f"[FATAL] All price entries NaN at idx={idx}")
            return None

        closes = closes_series[valid_mask]
        vol_series = vol_series[valid_mask]

        if len(closes) < 20:
            logger.warning(f"[SKIP] Not enough valid data to compute indicators at idx={idx} (len={len(closes)})")
            return None

        logger.debug(f"[INFO] Computing indicators at idx={idx} using closes from idx={start_idx} to idx={idx}")

        # --- EMA 20 ---
        ema_20 = closes.ewm(span=20).mean().iloc[-1]
        indicators["ema_20"] = ema_20

        # --- RSI 14 ---
        delta = closes.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=14).mean().iloc[-1]
        avg_loss = down.rolling(window=14).mean().iloc[-1] or 1e-6
        rs = avg_gain / avg_loss
        indicators["rsi_14"] = 100 - (100 / (1 + rs))

        # --- MACD ---
        exp1 = closes.ewm(span=12, adjust=False).mean()
        exp2 = closes.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators["macd"] = macd.iloc[-1]
        indicators["macd_signal"] = signal.iloc[-1]
        indicators["macd_hist"] = (macd - signal).iloc[-1]

        # --- Bollinger Bands (20) ---
        std = closes.rolling(window=20).std().iloc[-1]
        middle = closes.rolling(window=20).mean().iloc[-1]
        indicators["bb_middle"] = middle
        indicators["bb_upper"] = middle + 2 * std
        indicators["bb_lower"] = middle - 2 * std

        # --- VWAP ---
        window_clean = closes.values
        vol_clean = vol_series.values
        if len(window_clean) != len(vol_clean):
            logger.error(f"[FATAL] VWAP mismatch at idx={idx}: prices={len(window_clean)}, volumes={len(vol_clean)}")
            return None
        indicators["vwap"] = np.average(window_clean, weights=vol_clean)

        # --- ATR 14 (True range approximation) ---
        tr_list = [max(closes.iloc[i] - closes.iloc[i - 1], 0) for i in range(1, len(closes))]
        atr_14 = pd.Series(tr_list).rolling(window=14).mean().iloc[-1]
        indicators["atr_14"] = atr_14

        # --- Simulated ADX ---
        adx = RNG.uniform(10, 35)
        indicators["adx_14"] = adx

        # --- Final close price ---
        indicators["price"] = closes.iloc[-1] if not closes.empty else None

        # --- Final rounding ---
        for k in indicators:
            try:
                indicators[k] = round(float(indicators[k]), 4)
            except Exception as e:
                logger.error(f"[ERROR] Failed to round {k} at idx={idx}: {e}")
                indicators[k] = None

        logger.debug(f"[SUCCESS] Indicators at idx={idx}: {indicators}")
        return indicators

    except Exception as e:
        halt_on_error("compute_all_indicators", e, idx=idx, price_len=len(prices), volume_len=len(volumes))
        
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
        
def black_scholes_delta(s, k, t, r, sigma, call=True):
    if t <= 0:
        return 1.0 if call and s > k else 0.0 if call else -1.0 if s < k else 0.0
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return nd1 if call else nd1 - 1

def clean_bars(df):
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"clean_bars expected pd.DataFrame, got {type(df)}")

        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column in bars DataFrame: {col}")

        # Remove rows with any non-numeric or NaN in required fields
        df_clean = df.copy()
        for col in required_cols:
            df_clean = df_clean[pd.to_numeric(df_clean[col], errors="coerce").notna()]

        df_clean.reset_index(drop=True, inplace=True)

        logger.debug(f"🧽 clean_bars: cleaned bars from {len(df)} → {len(df_clean)} rows")
        return df_clean

    except Exception as e:
        halt_on_error("clean_bars", e, input_type=str(type(df)), input_preview=str(df.head() if isinstance(df, pd.DataFrame) else df))
        return pd.DataFrame()
        
def log_input_debug_info(entry_or_exit, trade_info, long_term_data, classifier_output, past_trades):
    logger.debug(f"\n==== {entry_or_exit} Meta-State DEBUG START ====")
    
    logger.debug(f"{entry_or_exit} | trade_info keys: {list(trade_info.keys())}")
    if 'entry_index' in trade_info:
        logger.debug(f"{entry_or_exit} | entry_index: {trade_info['entry_index']}")
    if 'exit_index' in trade_info:
        logger.debug(f"{entry_or_exit} | exit_index: {trade_info['exit_index']}")
    
    # --- long_term_data ---
    try:
        if hasattr(long_term_data, 'shape'):
            logger.debug(f"{entry_or_exit} | long_term_data shape: {long_term_data.shape}")
        elif isinstance(long_term_data, dict):
            logger.debug(f"{entry_or_exit} | long_term_data keys: {list(long_term_data.keys())}")
            for k, v in long_term_data.items():
                logger.debug(f"{entry_or_exit} | long_term_data[{k}]: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}, sample={str(v)[:200]}")
    except Exception as e:
        logger.error(f"{entry_or_exit} | Error with long_term_data: {e}")
    
    # --- classifier_output ---
    try:
        logger.debug(f"{entry_or_exit} | classifier_output keys: {list(classifier_output.keys())}")
        for k, v in classifier_output.items():
            logger.debug(f"{entry_or_exit} | classifier_output[{k}]: type={type(v)}, shape={np.shape(v)}, value={v if isinstance(v, (int, float)) else str(v)[:100]}")
    except Exception as e:
        logger.error(f"{entry_or_exit} | Error with classifier_output: {e}")
    
    # --- past_trades ---
    try:
        logger.debug(f"{entry_or_exit} | past_trades count: {len(past_trades)}")
        if isinstance(past_trades, list) and len(past_trades) > 0:
            logger.debug(f"{entry_or_exit} | sample past_trades[0]: {str(past_trades[0])[:300]}")
    except Exception as e:
        logger.error(f"{entry_or_exit} | Error with past_trades: {e}")

    logger.debug(f"==== {entry_or_exit} Meta-State DEBUG END ====\n")
    

def build_meta_state_for_entry(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    position_size: float = 0.0,
    trade_type: int = 1,
    confidence_score: float = 0.5,
    past_trades=None,
    long_term_data=None,
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    import pandas as pd
    import numpy as np
    import traceback

    past_trades = past_trades or []
    long_term_data = long_term_data or {}

    MIN_PAST_TRADES = 3  # enforce your required number here
    if len(past_trades) < MIN_PAST_TRADES:
        raise ValueError(f"🚫 Not enough past trades to build meta-state without padding (got {len(past_trades)}, need ≥ {MIN_PAST_TRADES})")

    logger.debug("🔍 Starting build_meta_state_for_entry")
    logger.debug(f"Types: 1m={type(data_1m)}, 5m={type(data_5m)}, 15m={type(data_15m)}, 1h={type(data_1h)}, 1d={type(data_1d)}")
    logger.debug(f"position_size={position_size}, trade_type={trade_type}, confidence_score={confidence_score}")
    logger.debug(f"past_trades={past_trades}")
    logger.debug(f"classifier_output={classifier_output}")

    # Validate long_term_data is a dict of DataFrames
    if not isinstance(long_term_data, dict):
        raise TypeError(f"❌ long_term_data must be a dict, got {type(long_term_data)}")
    for k, v in long_term_data.items():
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f"❌ long_term_data['{k}'] must be a DataFrame, got {type(v)}")

    logger.debug(f"✅ long_term_data keys: {list(long_term_data.keys())}")

    def ensure_df(df):
        if isinstance(df, dict):
            if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in df.values()):
                return pd.DataFrame(df)
            else:
                return pd.DataFrame([df])
        return df

    def build_sequence(state: List[float]) -> np.ndarray:
        padded = _pad(state)
        return np.stack([padded.copy() for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)

    # Ensure all incoming DataFrames are converted properly
    data_1m = ensure_df(data_1m)
    data_5m = ensure_df(data_5m)
    data_15m = ensure_df(data_15m)
    data_1h = ensure_df(data_1h)
    data_1d = ensure_df(data_1d)

    logger.debug(f"Data shapes: 1m={data_1m.shape}, 5m={data_5m.shape}, 15m={data_15m.shape}, 1h={data_1h.shape}, 1d={data_1d.shape}")

    try:
        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        logger.debug("✅ Fetched long-term ranges successfully.")

        def tf_feats(df):
            if df is None or len(df) == 0:
                logger.warning(f"⚠️ tf_feats: DataFrame is None or empty. Using defaults.")
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            try:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            except Exception as e:
                logger.error(f"❌ tf_feats error: {e}")
                return [PAD_VAL] * 4

        vix_val = fetch_vix_price() or 20.0
        logger.debug(f"VIX value: {vix_val}")

        if classifier_output and "regime_class" in classifier_output:
            regime = classifier_output["regime_class"]
        else:
            logger.debug("Classifying regime using _classify_regime()")
            regime = _classify_regime(data_1d.iloc[-1], vix_val)

        clf_conf = classifier_output.get("trade_success_prob") if classifier_output else None
        norm_conf = normalize(clf_conf if clf_conf is not None else confidence_score, DEFAULT_RANGES["CONF"])

        logger.debug(f"Normalized confidence: {norm_conf}, Regime: {regime}")

        state: List[float] = [
            norm_conf,
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            normalize(vix_val, DEFAULT_RANGES["VIX"]),
            normalize(position_size, DEFAULT_RANGES["SIZE"]),
            *_regime_one_hot(regime),
            *summarise_past(past_trades, prof_rng, dur_rng, DEFAULT_RANGES["SIZE"]),
            *tf_feats(data_1m), *tf_feats(data_5m),
            *tf_feats(data_15m), *tf_feats(data_1h), *tf_feats(data_1d),
        ]

        for p in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            df = long_term_data.get(p)
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    last = df.iloc[-1]
                    rsi_val = last.get("rsi", np.nan)
                    macd_val = last.get("macd", np.nan)
                    ema_dist = last.get("price", np.nan) - last.get("ema_20", np.nan)
    
                    state += [
                        normalize(rsi_val if pd.notna(rsi_val) else 50, rsi_rng),
                        normalize(macd_val if pd.notna(macd_val) else 0, macd_rng),
                        normalize(ema_dist if pd.notna(ema_dist) else 0, ema_rng),
                    ]
                except Exception as e:
                    logger.error(f"❌ Error extracting long_term_data[{p}]: {e}")
                    state += [PAD_VAL] * 3
            else:
                logger.warning(f"⚠️ Missing or empty long_term_data[{p}]")
                state += [PAD_VAL] * 3

        if classifier_output:
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            pred_dir = classifier_output.get("predicted_direction", -1)
            dir_one_hot = [0.0, 0.0, 0.0]
            if pred_dir in (0, 1, 2):
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL] * 3
            state.extend(dir_one_hot)

            class_probs = classifier_output.get("class_probabilities", [PAD_VAL] * 3)
            if not isinstance(class_probs, (list, tuple)) or len(class_probs) != 3:
                logger.warning(f"⚠️ Invalid class_probabilities format: {class_probs}")
                class_probs = [PAD_VAL] * 3
            else:
                class_probs = [float(x) if isinstance(x, (int, float)) else PAD_VAL for x in class_probs]
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            logger.debug("No classifier output provided, using PAD_VALs for classifier fields.")
            state += [PAD_VAL] * 8
        
        if len(state) < STATE_DIM:
            logger.warning(f"🚫 Meta state too short (length={len(state)}) — skipping")
            return None
        
        logger.debug("🧩 Component check before sequence build:")
        logger.debug(f" - len(state) = {len(state)} (expected ≥ {STATE_DIM})")
        logger.debug(f" - Last 10 elements of state: {state[-10:]}")
        logger.debug(f" - unique values: {set(np.round(state, 4))}")
        logger.debug(f" - tf_feats 1m: {tf_feats(data_1m)}")
        logger.debug(f" - tf_feats 5m: {tf_feats(data_5m)}")
        logger.debug(f" - tf_feats 15m: {tf_feats(data_15m)}")
        logger.debug(f" - tf_feats 1h: {tf_feats(data_1h)}")
        logger.debug(f" - tf_feats 1d: {tf_feats(data_1d)}")
        logger.debug(f" - classifier_output class_probs: {classifier_output.get('class_probabilities') if classifier_output else None}")

        result = build_sequence(state)
        if np.all(result == PAD_VAL):
            logger.warning("⚠️ Meta state is fully padded at entry — likely due to earlier data issue.")
        else:
            logger.debug(f"✅ Meta state successfully built. Shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"❌ Exception in build_meta_state_for_entry: {e}")
        logger.error(traceback.format_exc())
        return np.stack([_pad([PAD_VAL] * STATE_DIM) for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)
        
        
def build_meta_state_for_exit(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    position_size: float = 1.0,
    trade_type: int = 1,
    confidence_score: float = 0.5,
    entry_price: float = 0.0,
    final_price: float = 0.0,
    time_held_minutes: float = 0.0,
    past_trades=None,
    long_term_data=None,
    classifier_output: Optional[Dict] = None
) -> np.ndarray:
    import pandas as pd
    import numpy as np
    import traceback

    try:
        logger.debug("🚨 Building meta state for EXIT")
        logger.debug(f"🔹 Inputs: position_size={position_size}, trade_type={trade_type}, confidence_score={confidence_score}, entry_price={entry_price}, final_price={final_price}, time_held_minutes={time_held_minutes}")
        logger.debug(f"🔹 past_trades: {type(past_trades)}, len={len(past_trades) if past_trades else 0}")
        logger.debug(f"🔹 classifier_output keys: {list(classifier_output.keys()) if classifier_output else None}")

        past_trades = past_trades or []
        long_term_data = long_term_data or {}

        def ensure_df(df):
            if isinstance(df, dict):
                if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in df.values()):
                    return pd.DataFrame(df)
                else:
                    return pd.DataFrame([df])
            return df

        def build_sequence(state: List[float]) -> np.ndarray:
            padded = _pad(state)
            return np.stack([padded.copy() for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)

        data_1m = ensure_df(data_1m)
        data_5m = ensure_df(data_5m)
        data_15m = ensure_df(data_15m)
        data_1h = ensure_df(data_1h)
        data_1d = ensure_df(data_1d)

        logger.debug(f"✅ Timeframe shapes: 1m={data_1m.shape}, 5m={data_5m.shape}, 15m={data_15m.shape}, 1h={data_1h.shape}, 1d={data_1d.shape}")

        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df, tf_name="unknown"):
            if df is None:
                logger.warning(f"⚠️ {tf_name} df is None")
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            logger.warning(f"⚠️ {tf_name} df is empty or malformed")
            return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]

        vix_val = fetch_vix_price() or 20.0
        regime = classifier_output.get("regime_class") if classifier_output and "regime_class" in classifier_output else _classify_regime(data_1d.iloc[-1], vix_val)
        clf_conf = classifier_output.get("trade_success_prob") if classifier_output else None
        norm_conf = normalize(clf_conf if clf_conf is not None else confidence_score, DEFAULT_RANGES["CONF"])

        pnl_pct = (final_price - entry_price) / entry_price if entry_price > 0 else 0.0
        norm_pnl = normalize(pnl_pct, prof_rng)
        norm_time = normalize(time_held_minutes, dur_rng)

        state: List[float] = [
            norm_conf,
            1.0 if trade_type == 1 else 0.0,
            norm_time,
            normalize(vix_val, DEFAULT_RANGES["VIX"]),
            normalize(position_size, DEFAULT_RANGES["SIZE"]),
            *_regime_one_hot(regime),
            *summarise_past(past_trades, prof_rng, dur_rng, DEFAULT_RANGES["SIZE"]),
            norm_pnl,
            *tf_feats(data_1m, "1m"), *tf_feats(data_5m, "5m"),
            *tf_feats(data_15m, "15m"), *tf_feats(data_1h, "1h"), *tf_feats(data_1d, "1d"),
        ]

        for p in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            df = long_term_data.get(p)
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    last = df.iloc[-1]
                    rsi_val = last.get("rsi", np.nan)
                    macd_val = last.get("macd", np.nan)
                    ema_dist = last.get("price", np.nan) - last.get("ema_20", np.nan)
    
                    state += [
                        normalize(rsi_val if pd.notna(rsi_val) else 50, rsi_rng),
                        normalize(macd_val if pd.notna(macd_val) else 0, macd_rng),
                        normalize(ema_dist if pd.notna(ema_dist) else 0, ema_rng),
                    ]
                except Exception as e:
                    logger.error(f"❌ Error extracting long_term_data[{p}]: {e}")
                    state += [PAD_VAL] * 3
            else:
                logger.warning(f"⚠️ Missing or empty long_term_data[{p}]")
                state += [PAD_VAL] * 3

        if classifier_output:
            state.append(normalize(classifier_output.get("trade_success_prob", 0.5), DEFAULT_RANGES["CONF"]))
            pred_dir = classifier_output.get("predicted_direction", -1)
            dir_one_hot = [0.0, 0.0, 0.0]
            if pred_dir in (0, 1, 2):
                dir_one_hot[pred_dir] = 1.0
            else:
                dir_one_hot = [PAD_VAL] * 3
            state.extend(dir_one_hot)

            class_probs = classifier_output.get("class_probabilities", [PAD_VAL] * 3)
            logger.debug(f"🧠 class_probs={class_probs}")
            if not isinstance(class_probs, (list, tuple)) or len(class_probs) != 3:
                class_probs = [PAD_VAL] * 3
            else:
                class_probs = [float(x) if isinstance(x, (int, float)) else PAD_VAL for x in class_probs]
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if isinstance(entropy, (int, float)) and 0 <= entropy <= 1 else PAD_VAL)
        else:
            logger.warning("⚠️ classifier_output missing or None — padding final 8 features.")
            state += [PAD_VAL] * 8

        if len(state) < STATE_DIM:
            logger.warning(f"🚫 Meta EXIT state too short (length={len(state)}) — skipping")
            return None
            
        result = build_sequence(state)
        if np.all(result == PAD_VAL):
            logger.error("❌ Meta state for EXIT is fully padded! Likely due to bad input.")
        logger.debug(f"✅ Final EXIT meta state shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"❌ Exception in build_meta_state_for_exit: {e}")
        logger.error(traceback.format_exc())
        return np.stack([_pad([PAD_VAL] * STATE_DIM) for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)

def simulate_trade(day, trade_idx, closes, volumes, vix_shift, long_term_data, meta_stats: dict):
    logger.debug(f"🚀 Starting simulate_trade | Day: {day}, Trade Index: {trade_idx}")

    max_required = 50000

    if len(closes) < 2000:
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: not enough 1m bars (len(closes)={len(closes)})")
        return None

    start_idx = RNG.randint(60, len(closes) - 60)
    trade_minute = start_idx
    logger.debug(f"📍 Chosen trade_minute={trade_minute} (from index range 60 to {len(closes) - 60})")

    required_lookback = {
        "1m": 60,
        "5m": 150,
        "15m": 300,
        "1h": 600,
        "1d": 60,
    }

    indicator_buffer = {
        "1m": 50,
        "5m": 50,
        "15m": 50,
        "1h": 50,
        "1d": 50,
    }

    input_slices = {
        "1m": required_lookback["1m"] + indicator_buffer["1m"],
        "5m": (required_lookback["5m"] + indicator_buffer["5m"]) * 5,
        "15m": (required_lookback["15m"] + indicator_buffer["15m"]) * 15,
        "1h": (required_lookback["1h"] + indicator_buffer["1h"]) * 60,
        "1d": (required_lookback["1d"] + indicator_buffer["1d"]) * 390,
    }

    min_required_minutes = max(input_slices.values())
    logger.debug(f"📏 Min required bars across timeframes: {min_required_minutes} minutes")

    if trade_minute < min_required_minutes:
        logger.debug(
            f"⏩ Skipping trade {trade_idx} on day {day}: "
            f"trade_minute={trade_minute} < min_required_minutes={min_required_minutes}"
        )
        return None

    total_offset = timedelta(days=day, minutes=trade_minute)
    base_time = datetime(2025, 1, 1, 9, 30) + total_offset

    try:
        logger.debug(f"🔧 Constructing bars with slices ending at trade_minute={trade_minute}")

        bars_1m = construct_bars(
            closes[trade_minute - input_slices["1m"]:trade_minute],
            volumes[trade_minute - input_slices["1m"]:trade_minute],
            1,
            start_time=base_time - timedelta(minutes=input_slices["1m"] - 1)
        )
        logger.debug(f"✅ bars_1m: shape={bars_1m.shape} | last close={bars_1m.iloc[-1]['close']:.2f}")

        bars_5m = construct_bars(
            closes[trade_minute - input_slices["5m"]:trade_minute],
            volumes[trade_minute - input_slices["5m"]:trade_minute],
            5,
            start_time=base_time - timedelta(minutes=input_slices["5m"] - 5)
        )
        logger.debug(f"✅ bars_5m: shape={bars_5m.shape} | last close={bars_5m.iloc[-1]['close']:.2f}")

        bars_15m = construct_bars(
            closes[trade_minute - input_slices["15m"]:trade_minute],
            volumes[trade_minute - input_slices["15m"]:trade_minute],
            15,
            start_time=base_time - timedelta(minutes=input_slices["15m"] - 15)
        )
        logger.debug(f"✅ bars_15m: shape={bars_15m.shape} | last close={bars_15m.iloc[-1]['close']:.2f}")

        bars_1h = construct_bars(
            closes[trade_minute - input_slices["1h"]:trade_minute],
            volumes[trade_minute - input_slices["1h"]:trade_minute],
            60,
            start_time=base_time - timedelta(minutes=input_slices["1h"] - 60)
        )
        logger.debug(f"✅ bars_1h: shape={bars_1h.shape} | last close={bars_1h.iloc[-1]['close']:.2f}")

        bars_1d = construct_bars(
            closes[trade_minute - input_slices["1d"]:trade_minute],
            volumes[trade_minute - input_slices["1d"]:trade_minute],
            390,
            start_time=base_time - timedelta(minutes=input_slices["1d"] - 390)
        )
        logger.debug(f"✅ bars_1d: shape={bars_1d.shape} | last close={bars_1d.iloc[-1]['close']:.2f}")

    except Exception as e:
        logger.exception(f"❌ Exception during bar construction for trade {trade_idx} on day {day}: {e}")
        return None

    # 🧹 Clean bars and log pre/post cleaning info
    for tf_name, bars in zip(["1m", "5m", "15m", "1h", "1d"], [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        logger.debug(f"🧼 Cleaning bars for {tf_name}... pre-clean len={len(bars)}")
        
    bars_1m = clean_bars(bars_1m)
    bars_5m = clean_bars(bars_5m)
    bars_15m = clean_bars(bars_15m)
    bars_1h = clean_bars(bars_1h)
    bars_1d = clean_bars(bars_1d)
    
    # ✅ Validate bar lengths
    required_bars = required_lookback
    for tf, bars in zip(required_bars, [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        if not isinstance(bars, pd.DataFrame):
            logger.error(f"❌ Bars for {tf} is not a DataFrame (type={type(bars)}) — skipping trade {trade_idx} on day {day}")
            return None
        if len(bars) < required_bars[tf]:
            logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: only {len(bars)} bars for {tf} (required: {required_bars[tf]})")
            return None
        logger.debug(f"✅ {tf} bars OK: len={len(bars)} (required={required_bars[tf]})")
    
    # 📊 OHLCV array construction from 1m bars
    try:
        closes_1m = bars_1m["close"].tolist()
        opens_1m = bars_1m["open"].tolist()
        highs_1m = bars_1m["high"].tolist()
        lows_1m = bars_1m["low"].tolist()
        volumes_1m = bars_1m["volume"].tolist()
    except Exception as e:
        logger.exception(f"❌ Failed to construct OHLCV arrays from bars_1m: {e}")
        return None
    
    logger.debug(f"📊 OHLCV arrays extracted from 1m bars:")
    logger.debug(f"   • closes_1m[0:3]: {closes_1m[:3]} ... len={len(closes_1m)}")
    logger.debug(f"   • opens_1m[0:3]: {opens_1m[:3]} ... len={len(opens_1m)}")
    logger.debug(f"   • highs_1m[0:3]: {highs_1m[:3]} ... len={len(highs_1m)}")
    logger.debug(f"   • lows_1m[0:3]: {lows_1m[:3]} ... len={len(lows_1m)}")
    logger.debug(f"   • volumes_1m[0:3]: {volumes_1m[:3]} ... len={len(volumes_1m)}")
    
    # 🧱 Entry window validation
    bars_by_tf = {
        "1m": bars_1m,
        "5m": bars_5m,
        "15m": bars_15m,
        "1h": bars_1h,
        "1d": bars_1d  # assuming daily bars are stored in this variable
    }
    
    for tf, bars in bars_by_tf.items():
        if len(bars) < required_bars[tf]:
            logger.debug(
                f"⏩ Skipping trade {trade_idx} on day {day}: "
                f"len(bars_{tf}) ({len(bars)}) < required_bars['{tf}'] ({required_bars[tf]})"
            )
            return None
    
    max_start_idx = len(bars_1m) - required_bars["1m"]
    logger.debug(f"🧮 max_start_idx = {max_start_idx} (bars_1m={len(bars_1m)} - required={required_bars['1m']})")
    
    if max_start_idx < 0:
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: max_start_idx={max_start_idx} < 0")
        return None
    
    start_idx = RNG.randint(0, max_start_idx)
    logger.debug(f"🎯 Chosen start_idx={start_idx} within range [0, {max_start_idx}]")
    
    # 💰 Option strike and pricing setup
    price_sig = closes_1m[start_idx]
    logger.debug(f"📈 Price signal at start_idx={start_idx}: {price_sig:.2f}")
    
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    logger.debug(f"🎯 Generated strike price: {strike:.2f}")
    
    option_type = RNG.choice(["C", "P"])
    logger.debug(f"🔀 Selected option type: {option_type}")
    
    expiry_days = RNG.randint(7, 30)
    t_expiry = expiry_days / 365
    logger.debug(f"🗓️ Option expiry_days={expiry_days}, t_expiry={t_expiry:.4f} years")
    
    day_dt = base_time + timedelta(days=day)
    option_symbol = make_option_symbol(day_dt, strike, option_type)
    logger.debug(f"🏷️ Option symbol: {option_symbol}")
    
    # 📉 Option price + delta via Black-Scholes
    try:
        option_price = black_scholes_price(
            s=price_sig,
            k=strike,
            t=t_expiry,
            r=0.01,
            sigma=0.25,
            call=(option_type == "C")
        )
    
        option_delta = black_scholes_delta(
            s=price_sig,
            k=strike,
            t=t_expiry,
            r=0.01,
            sigma=0.25,
            call=(option_type == "C")
        )
    
        option_data = {
            "price": option_price,
            "iv": 0.25,
            "delta": option_delta
        }
        logger.debug(f"📊 Black-Scholes price: {option_price:.4f}, delta: {option_delta:.4f}")
    
    except Exception as e:
        logger.error(f"❌ Black-Scholes error: {e}")
        option_data = {
            "price": 0.0,
            "iv": 0.25,
            "delta": 0.0
        }
    
    # 🎯 Slippage, fill %, entry price
    slippage = RNG.uniform(-0.5, 0.5) / 100
    fill_pct = RNG.uniform(0.7, 1.0)
    entry_price = round(option_price * (1 + slippage), 2)
    logger.debug(f"💸 Slippage: {slippage*100:.2f}%, fill_pct: {fill_pct:.3f}, entry_price: {entry_price:.2f}")
    
    # 🌀 Swing trade randomization
    is_swing = RNG.random() < 0.2
    logger.debug(f"🔄 is_swing: {is_swing}")
    
    if len(bars_1m) < 101:
        logger.debug("❌ Not enough bars for indicators (need ≥101). Skipping trade.")
        return None
    
    # 🧠 Compute indicators with strict validation
    try:
        indicators = compute_all_indicators(closes_1m, volumes_1m, len(closes_1m) - 1)
        required_keys = [
            'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'vwap', 'atr_14', 'adx_14'
        ]
        bad_keys = []
        for key in required_keys:
            val = indicators.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                bad_keys.append(key)
    
        if bad_keys:
            logger.error(f"🚫 Indicator computation failed or returned NaN for keys: {bad_keys}")
            logger.debug(f"🧪 Full indicator output:\n{indicators}")
            logger.debug(f"📉 closes_1m[-5:]: {closes_1m[-5:].tolist()}")
            return None
        else:
            logger.debug(f"🧠 Indicators valid at {len(bars_1m) - 1}: {indicators}")
    except Exception as e:
        logger.exception("❌ Exception during indicator computation")
        return None
    
    # 🔮 Classifier features
    classifier_confidence = round(np.random.beta(5, 2), 2)
    setup_quality = round(RNG.uniform(0.6, 1.0), 2)
    vix = round(RNG.uniform(15, 35), 2)
    
    logger.debug(f"🧪 Classifier confidence: {classifier_confidence}")
    logger.debug(f"🧰 Setup quality: {setup_quality}")
    logger.debug(f"🌪️ Simulated VIX: {vix}")
    
    if start_idx >= 20:
        realized_vol = round(np.std(closes_1m[start_idx - 20:start_idx]), 2)
        logger.debug(f"📉 Realized vol over 20-bar window: {realized_vol}")
    else:
        realized_vol = 1.5
        logger.debug("ℹ️ Not enough bars for realized_vol — defaulting to 1.5")
    
    trade_type = 0 if option_type == "C" else 1
    logger.debug(f"📦 Encoded trade_type: {trade_type} (0=Call, 1=Put)")
    
    total_signals_today = RNG.randint(0, 10)
    logger.debug(f"📊 Simulated total_signals_today: {total_signals_today}")
    
        # ============================
    # Classifier feature construction
    # ============================
    try:
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
        logger.debug(f"✅ classifier_features constructed: keys={list(classifier_features.keys())}")
      # 🔍 Sanity check for non-scalar values
        for k, v in classifier_features.items():
            if isinstance(v, (list, np.ndarray)) and len(v) != 1:
                logger.warning(f"⚠️ classifier_feature '{k}' might be non-scalar: {v}")
    except KeyError as ke:
        logger.error(f"❌ Missing key in indicators while building classifier_features: {ke}")
        return None
    except Exception as e:
        logger.exception("❌ Exception during classifier_features construction")
        return None

  # ✅ Forced acceptance of dummy past trades for debugging
    past_trades_raw = TRADE_HISTORY[-15:] if len(TRADE_HISTORY) >= 15 else TRADE_HISTORY
    past_trades = []
    
    # 💥 Skip strict validation and force usage
    for i, t in enumerate(past_trades_raw):
        if isinstance(t, dict):
            past_trades.append(t)
        else:
            logger.warning(f"🛑 Skipped invalid past_trade[{i}] — Not a dict, type: {type(t)}")
    
    # 🧱 Ensure we have at least 3 past trades by padding with dummy data if needed
    MIN_PAST_TRADES = 3
    if len(past_trades) < MIN_PAST_TRADES:
        logger.warning(f"⚠️ Only {len(past_trades)} past_trades found. Padding with dummy trades...")
        dummy_trade = {
            'pct_pnl': 0.0,
            'duration': 1,
            'position_size': 1,
            'classifier_output': [0.33, 0.33, 0.33],
            'classifier_features': {f'feat_{i}': 0.0 for i in range(30)},
            'meta_entry': [0.5]*64,
            'meta_exit': [0.5]*64,
            'option_data': {},
            'indicators': {}
        }
        while len(past_trades) < MIN_PAST_TRADES:
            past_trades.append(dummy_trade)
    
    logger.debug(f"✅ Using {len(past_trades)} past_trades (with padding if needed) for meta-state")
    
    # ✅ OPTIONAL: If you need the *latest* classifier_features for next-step processing
    try:
        if past_trades:
            latest_features = past_trades[-1]["classifier_features"]
            features_df = build_features_for_trade(latest_features)
            logger.debug(f"🛠️ Built features_df: type={type(features_df)}, shape={getattr(features_df, 'shape', 'N/A')}")
        else:
            logger.warning("⚠️ No valid past_trades to extract classifier_features from.")
            features_df = None
    except Exception as e:
        logger.exception("❌ Failed to build features_df")
        features_df = None

    # Update long-term stats
    try:
        update_long_term_stats(long_term_data, classifier_features)
    except Exception as e:
        logger.exception("❌ Failed to update long_term_data")
        return None
    
    # Validate long_term_data integrity
    for k, v in long_term_data.items():
        if not isinstance(v, pd.DataFrame):
            logger.error(f"❌ long_term_data[{k}] is not a DataFrame (got {type(v)})")
            raise TypeError(f"Invalid long_term_data[{k}]")
    
    # 🔒 Ensure required long-term keys are populated
    for horizon in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
        if horizon not in long_term_data or long_term_data[horizon].empty:
            logger.warning(f"⚠️ Missing or empty long_term_data[{horizon}] — inserting fallback")
            long_term_data[horizon] = pd.DataFrame([classifier_features])
    
    # Ensure proper DataFrame for classifier input
    if not isinstance(features_df, pd.DataFrame):
        logger.warning("⚠️ build_features_for_trade did not return DataFrame, coercing...")
        features_df = pd.DataFrame([classifier_features])

    if features_df.shape[0] != 1:
        logger.error(f"🚫 Invalid features_df shape: {features_df.shape} (expected 1 row)")
        logger.debug(f"🧬 Full features_df:\n{features_df}")
        logger.debug(f"📦 Raw classifier_features:\n{classifier_features}")
        logger.debug(f"🕒 Trade {trade_idx} on day {day}")
        raise ValueError(f"Invalid features_df shape: {features_df.shape}")

    try:
        inference = ModelInference()
        raw_output = inference.predict_with_confidence(features_df)
        classifier_output = ModelInference.wrap_classifier_output(raw_output)
        logger.debug(f"🧠 Classifier output: {classifier_output}")
        
        confidence = classifier_output.get("trade_success_prob", 0.5)
        entropy = classifier_output.get("entropy", 1.0)
        trade_success_prob = classifier_output.get("trade_success_prob", 0.5)
        class_probabilities = classifier_output.get("class_probabilities", [0.5, 0.5, 0.0])
    except Exception as e:
        logger.exception("❌ Model inference failed")
        return None
    
    # Confidence-based position sizing
    position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * confidence
    logger.debug(f"📏 Position size based on confidence {confidence:.2f}: {position_size:.4f}")
        
    # 🔍 DEBUG: Log everything going into build_meta_state_for_entry
    logger.debug(f"🧱 Calling build_meta_state_for_entry with keys: {[k for k in locals().keys()]}")
    logger.debug(f"Types: bars_1m={type(bars_1m)}, past_trades={type(past_trades)}, long_term_data={type(long_term_data)}")
    logger.debug(f"Classifier output keys: {list(classifier_output.keys())}")
    logger.debug(f"Lengths: 1m={len(bars_1m)}, 5m={len(bars_5m)}, 15m={len(bars_15m)}, 1h={len(bars_1h)}, 1d={len(bars_1d)}")
    
    # ✅ Validate long_term_data BEFORE using it in meta-state functions
    if not isinstance(long_term_data, dict):
        raise TypeError(f"❌ long_term_data must be a dict, got {type(long_term_data)}")
    
    for key in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
        if key not in long_term_data or long_term_data[key] is None or long_term_data[key].empty:
            logger.warning(f"⚠️ Missing or empty long_term_data[{key}]")
            
    # 🧠 Log all inputs to build_meta_state_for_entry()
    logger.debug("🔍 Logging inputs to build_meta_state_for_entry()")
    
    for tf_name, bars in {
        "1m": bars_1m,
        "5m": bars_5m,
        "15m": bars_15m,
        "1h": bars_1h,
        "1d": bars_1d
    }.items():
        logger.debug(f"Bars {tf_name}: type={type(bars)}, shape={getattr(bars, 'shape', 'N/A')}")
        if isinstance(bars, pd.DataFrame):
            logger.debug(f"Bars {tf_name} head:\n{bars.head(2)}")
    
    logger.debug(f"classifier_output type: {type(classifier_output)}")
    if isinstance(classifier_output, dict):
        for k, v in classifier_output.items():
            logger.debug(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}, value={v if isinstance(v, (int, float)) else str(v)[:200]}")
    
    logger.debug(f"long_term_data type: {type(long_term_data)}")
    if isinstance(long_term_data, dict):
        for k, v in long_term_data.items():
            logger.debug(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}, value preview={str(v)[:200]}")
    
    logger.debug(f"past_trades type: {type(past_trades)}, length: {len(past_trades)}")
    if past_trades:
        logger.debug(f"First past_trade keys: {list(past_trades[0].keys()) if isinstance(past_trades[0], dict) else type(past_trades[0])}")
        
    # 🔒 Validate classifier_output before building meta-state
    required_classifier_keys = ["trade_success_prob", "entropy", "class_probabilities"]
    missing_cls_keys = [k for k in required_classifier_keys if k not in classifier_output or classifier_output[k] is None]
    
    if missing_cls_keys:
        logger.error(f"❌ classifier_output is missing or None for keys: {missing_cls_keys}")
        raise ValueError(f"Invalid classifier_output: missing {missing_cls_keys}")
    
    for k in ["trade_success_prob", "entropy"]:
        if not isinstance(classifier_output[k], (float, int)):
            logger.error(f"❌ classifier_output['{k}'] must be numeric, got {type(classifier_output[k])}")
            raise TypeError(f"Invalid type for classifier_output['{k}']")
    
    if not isinstance(classifier_output.get("class_probabilities", []), (list, np.ndarray)):
        logger.error("❌ classifier_output['class_probabilities'] is not list or array")
        raise TypeError("Invalid type for class_probabilities")
        
        # 🔍 DEBUG BEFORE build_meta_state_for_entry
    try:
        log_input_debug_info(
            entry_or_exit="ENTRY",
            trade_info={"start_idx": start_idx, "trade_idx": trade_idx, "day": day},  # Simplified
            long_term_data=long_term_data,
            classifier_output=classifier_output,
            past_trades=past_trades
        )
    except Exception as dbg_e:
        logger.exception("❌ log_input_debug_info failed before meta_entry")
                
    # Build meta-entry
    try:
        meta_entry = build_meta_state_for_entry(
            data_1m=bars_1m,
            data_5m=bars_5m,
            data_15m=bars_15m,
            data_1h=bars_1h,
            data_1d=bars_1d,
            position_size=position_size,
            confidence_score=classifier_confidence,
            trade_type=int(is_swing),
            past_trades=past_trades,
            long_term_data=long_term_data,
            classifier_output=classifier_output
        )
        logger.debug(f"🧩 meta_entry shape: {np.array(meta_entry).shape if meta_entry is not None else 'None'}")
    
    except Exception as e:
        logger.exception("❌ Failed to build meta_entry")
        return None
    
    if meta_entry is None:
        meta_stats["empty_states"] += 1
        logger.error(f"🚫 meta_entry is None — trade {trade_idx} on day {day} cannot proceed.")
        return None
    
    if is_padded(meta_entry):
        meta_stats["empty_states"] += 1
        logger.error(f"🚫 Padded meta_entry detected for trade {trade_idx} on day {day} — aborting.")
        try:
            meta_np = np.array(meta_entry)
            logger.error(f"🧩 meta_entry shape: {meta_np.shape}")
            logger.error(f"🧩 meta_entry preview (first 10): {meta_np.flatten()[:10].tolist()}")
            logger.error(f"🧩 meta_entry unique values: {np.unique(meta_np)}")
        except Exception as log_err:
            logger.exception("❌ Failed to log meta_entry structure")
        return None
    
    if classifier_output is None:
        meta_stats["no_classifier"] += 1
        logger.error(f"🚫 classifier_output is None — trade {trade_idx} on day {day}")
        return None
    
    # Meta agent action
    try:
        action, agent_confidence = meta_agent.select_action(meta_entry)
        logger.debug(f"🎯 Meta-agent action={action}, confidence={agent_confidence:.2f}")
        meta_stats["decisions_made"] += 1
    except Exception as e:
        logger.exception("❌ Meta-agent select_action failed")
        return None
    
    if action == 0:
        meta_stats["other_skips"] += 1
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: meta-agent skipped (action=0)")
        return None
    
    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    logger.debug(f"📈 Planned trade duration: {duration} (swing={is_swing})")

    if start_idx + duration >= len(closes_1m):
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: duration exceeds data (start_idx={start_idx}, len={len(closes_1m)})")
        return None

    final_price = closes_1m[start_idx + duration]
    logger.debug(f"📉 Final price at exit: {final_price:.2f}")

    minutes_per_year = 252 * 6.5 * 60
    time_left = max(t_expiry - (duration / minutes_per_year), 0.01)
    logger.debug(f"⏳ Time to expiry after holding: {time_left:.4f} years")

    try:
        new_option_price = black_scholes_price(
            s=final_price,
            k=strike,
            t=time_left,
            r=0.01,
            sigma=0.25,
            call=(option_type == "C")
        )
    except Exception as e:
        logger.exception("❌ Error computing exit option price")
        return None

    exit_price = round(new_option_price * (1 + slippage), 2)
    logger.debug(f"💸 Slippage-adjusted exit price: {exit_price:.2f}")

    gross_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * fill_pct
    total_commission = 2 * COMMISSION_PER_CONTRACT
    raw_pnl = gross_pnl - total_commission
    initial_cost = entry_price * CONTRACT_MULTIPLIER * fill_pct + 1e-9  # Avoid division by zero
    pct_pnl = (raw_pnl / initial_cost) * 100
    trade_result = pct_pnl

    logger.debug(f"📊 Trade PnL details:")
    logger.debug(f"   • Entry price: {entry_price}, Exit price: {exit_price}")
    logger.debug(f"   • Gross PnL: {gross_pnl:.2f}, Raw PnL: {raw_pnl:.2f}, % PnL: {pct_pnl:.2f}%")

    atr = indicators.get('atr_14', 1.0)
    logger.debug(f"📐 ATR(14): {atr:.2f}")
    
    # 🏁 Build exit meta-state
    try:
        logger.debug(f"🧱 Calling build_meta_state_for_exit with keys: {[k for k in locals().keys()]}")
        logger.debug(f"Types: bars_1m={type(bars_1m)}, past_trades={type(past_trades)}, long_term_data={type(long_term_data)}")
        logger.debug(f"Classifier output keys: {list(classifier_output.keys())}")
    
        try:
            log_input_debug_info(
                entry_or_exit="EXIT",
                trade_info={
                    "exit_price": exit_price,
                    "duration": duration,
                    "trade_idx": trade_idx,
                    "day": day
                },
                long_term_data=long_term_data,
                classifier_output=classifier_output,
                past_trades=past_trades
            )
        except Exception as dbg_e:
            logger.exception("❌ log_input_debug_info failed before meta_exit")
    
        meta_exit = build_meta_state_for_exit(
            data_1m=bars_1m,
            data_5m=bars_5m,
            data_15m=bars_15m,
            data_1h=bars_1h,
            data_1d=bars_1d,
            confidence_score=agent_confidence,
            trade_type=int(is_swing),
            entry_price=entry_price,
            final_price=exit_price,
            time_held_minutes=duration,
            past_trades=past_trades,
            long_term_data=long_term_data,
            classifier_output=classifier_output
        )
        logger.debug(f"🧩 meta_exit shape: {np.array(meta_exit).shape if meta_exit is not None else 'None'}")
    
    except Exception as e:
        logger.exception("❌ Failed to build meta_exit")
        return None
    
    if meta_exit is None:
        meta_stats["empty_states"] += 1
        logger.debug(f"🚫 Skipping trade {trade_idx} on day {day}: meta_exit returned None")
        return None
    
    if is_padded(meta_exit):
        meta_stats["empty_states"] += 1
        logger.debug(f"🚫 Skipping trade {trade_idx} on day {day}: meta_exit is padded")
        return None
    
    # 🧠 Track trade history for future states
    if trade_result is not None and isinstance(classifier_features, dict):
        TRADE_HISTORY.append({
            "pct_pnl": trade_result,
            "duration": duration,
            "position_size": position_size,
            "classifier_features": classifier_features,
        })
        logger.debug(f"✅ Trade appended to TRADE_HISTORY — keys: {TRADE_HISTORY[-1].keys()}")
    else:
        logger.warning(f"❌ Skipped appending invalid trade — trade_result: {trade_result}, classifier_features: {type(classifier_features)}")
        
    predicted_direction = classifier_output.get("predicted_direction", -1)
    
    # 🎯 Validate direction prediction
    direction_correct = (
        (predicted_direction == 1 and final_price > price_sig) or
        (predicted_direction == 0 and final_price < price_sig)
    )
    logger.debug(f"🎯 Trade {trade_idx} on day {day}: Direction predicted={predicted_direction}, Actual={final_price:.2f} vs Signal={price_sig:.2f} → Correct={direction_correct}")
    logger.debug(f"📈 Trade result before reward shaping: {trade_result}")
    # 🏆 Compute shaped reward
    try:
        shaped_reward = reward_shaper.compute_shaped_reward(
            trade_result={
                "pct_pnl": trade_result,
                "setup_quality": setup_quality,
                "entry_quality": abs(trade_result) / max(atr, 1e-6),
                "direction_correct": direction_correct,
                "trades_today": trade_idx,
                "was_successful": trade_result > 0,
                "risk_reward_ratio": abs(trade_result) / max(atr, 1e-6),
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
    except Exception as e:
        logger.error(f"❌ Failed to compute shaped reward: {e}")
        return None
    
    logger.debug(f"🏅 Trade {trade_idx} on day {day}: Shaped reward = {shaped_reward:.2f}")
    
    # 🗑️ Garbage filtering
    if shaped_reward < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        logger.debug(f"🗑️ Skipping trade {trade_idx} on day {day}: shaped_reward too low ({shaped_reward:.2f})")
        return None
    
    # 🧪 Check bounds before accessing bars_1m
    if start_idx >= len(bars_1m):
        logger.error(f"❌ start_idx={start_idx} out of bounds for bars_1m with length={len(bars_1m)}")
        return None
        
    # 📝 Timestamp conversion
    try:
        entry_bar = bars_1m.iloc[start_idx]  # ✅ FIXED
    except Exception as e:
        logger.exception(f"❌ Failed to access entry_bar at start_idx={start_idx} in bars_1m")
        return None
    
    ts = entry_bar.get("timestamp", None)
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts)
    elif not isinstance(ts, datetime):
        logger.warning(f"⚠️ Unexpected timestamp type: {type(ts)} — defaulting to now()")
        ts = datetime.now()
    
    # 💾 Log training example
    try:
        log_training_example(
            timestamp=ts,
            close=entry_bar.get("close", 0),
            features=classifier_features,
            label=trade_result,
            meta_entry_state=meta_entry.tolist(),
            meta_exit_state=meta_exit.tolist(),
        )
        logger.debug(f"✅ Trade {trade_idx} logged | PnL={trade_result:.2f}% | Option={option_symbol} | Duration={duration}m")
    except Exception as e:
        logger.error(f"❌ Failed to log training example: {e}")
        return None
    
    # 🔍 Final shape/type safety for meta states
    if not isinstance(meta_entry, np.ndarray):
        logger.warning(f"⚠️ meta_entry not ndarray (type={type(meta_entry)}), coercing")
        meta_entry = np.array(meta_entry)
    
    if not isinstance(meta_exit, np.ndarray):
        logger.warning(f"⚠️ meta_exit not ndarray (type={type(meta_exit)}), coercing")
        meta_exit = np.array(meta_exit)
    
    # 📦 Final trade result dict
    trade = {
        "timestamp": str(ts),
        "day": day,
        "trade_idx": trade_idx,
        "option": option_symbol,
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
        "indicators": indicators,
        "option_data": option_data,
        "position_size": position_size,
    }
    
    # ✅ Log full trade to meta log (robust version)
    try:
        write_to_meta_log(trade)
    except Exception as e:
        logger.error(f"❌ Failed to write trade to meta log: {e}")
        traceback.print_exc()
    
    # ✅ Final validation before returning
    required_keys = ["entry_price", "exit_price", "pct_pnl", "meta_entry", "meta_exit"]
    if all(k in trade and trade[k] is not None for k in required_keys):
        logger.debug("✅ Trade dict complete — returning trade")
        return trade
    else:
        logger.warning(f"🚫 Incomplete trade dict — missing keys: {[k for k in required_keys if k not in trade or trade[k] is None]}")
        return None
    
def main():
    global ACCUMULATED_CLOSES, ACCUMULATED_VOLUMES
    
    # ✅ Initialize meta-agent diagnostics
    meta_stats = {
        "decisions_made": 0,
        "empty_states": 0,
        "no_classifier": 0,
        "other_skips": 0
    }

     #=== Inject dummy trades before simulation starts ===
    if len(TRADE_HISTORY) < 10:
        logger.debug("🤖 Injecting 10 dummy trades into TRADE_HISTORY to bootstrap simulation")
        for _ in range(10):
            TRADE_HISTORY.append(generate_dummy_trade())
            
    for day in range(SIM_DAYS):
        try:
            is_warmup = day < WARM_UP_DAYS
    
            # === Generate price and volume for the day ===
            daily_prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
            daily_volumes = [RNG.randint(300_000, 1_000_000) for _ in daily_prices]
    
            ACCUMULATED_CLOSES.extend(daily_prices)
            ACCUMULATED_VOLUMES.extend(daily_volumes)
    
            # === Handle warm-up days ===
            if is_warmup:
                logger.debug(f"🌱 Warm-up Day {day + 1}: Populating market buffers")
                vix_shift = RNG.uniform(14, 28)
    
                for trade_idx in range(min(2, TRADES_PER_DAY)):
                    try:
                        logger.debug(f"🧪 Attempting warm-up trade {trade_idx + 1} on Day {day + 1}")
                        log_entry = simulate_trade(
                            day,
                            trade_idx,
                            ACCUMULATED_CLOSES,
                            ACCUMULATED_VOLUMES,
                            vix_shift,
                            long_term_data
                        )
    
                        if log_entry:
                            TRADE_HISTORY.append(log_entry)
                            logger.debug(f"✅ Warm-up Trade {trade_idx + 1} appended to TRADE_HISTORY: total={len(TRADE_HISTORY)}")
                        else:
                            logger.debug(f"❌ Warm-up trade {trade_idx + 1} failed (simulate_trade returned None)")
                    except Exception:
                        logger.warning(f"⚠️ Exception in warm-up simulate_trade() for trade {trade_idx + 1}")
                        traceback.print_exc()
    
                if day == WARM_UP_DAYS - 1:
                    logger.debug(f"📦 TRADE_HISTORY after warm-up: {len(TRADE_HISTORY)} trades collected")
                continue  # Skip main logic during warm-up
    
            # ✅ Inject dummy trades once after warm-up
            if day == WARM_UP_DAYS and len(TRADE_HISTORY) == 0:
                for _ in range(10):
                    TRADE_HISTORY.append(generate_dummy_trade())
                logger.info("🧪 Injected 10 dummy trades on day 110 to seed trade history")
    
            # === Main simulation logic ===
            vix_shift = RNG.uniform(14, 28)
            if RNG.random() < 0.08:
                vix_shift += RNG.uniform(5, 15)

            trades = []
            successful_trades = 0
            logger.debug(f"📈 Day {day + 1}: Starting simulation with VIX shift {vix_shift:.2f} | Total prices: {len(ACCUMULATED_CLOSES)}")

            for trade_idx in range(TRADES_PER_DAY):
                try:
                    logger.debug(f"📜 TRADE_HISTORY length: {len(TRADE_HISTORY)} before trade {trade_idx + 1}")

                    if len(TRADE_HISTORY) < 3:
                        logger.debug(f"🟡 TRADE_HISTORY has only {len(TRADE_HISTORY)} trades — allowing early trades to accumulate")
                        log_entry = simulate_trade(day, trade_idx, ACCUMULATED_CLOSES, ACCUMULATED_VOLUMES, vix_shift, long_term_data)
                        if log_entry:
                            trades.append(log_entry)
                            TRADE_HISTORY.append(log_entry)
                            logger.debug(f"✅ Trade appended to TRADE_HISTORY: total={len(TRADE_HISTORY)}")
                        continue  # Skip meta-agent logic

                    log_entry = simulate_trade(day, trade_idx, ACCUMULATED_CLOSES, ACCUMULATED_VOLUMES, vix_shift, long_term_data, meta_stats)

                    if log_entry:
                        trades.append(log_entry)
                        TRADE_HISTORY.append(log_entry)
                        logger.debug(f"✅ Trade appended to TRADE_HISTORY: total={len(TRADE_HISTORY)}")
                        successful_trades += 1

                        logger.debug(f"✅ Trade {trade_idx + 1} | PnL={log_entry['pct_pnl']}% | Duration={log_entry['duration']} mins")

                        indicators = log_entry.get("indicators", {})
                        option_data = log_entry.get("option_data", {})
                        position_size = log_entry.get("position_size", 1.0)

                        def append_if_valid(key, val):
                            if val is not None and isinstance(val, (int, float)) and not math.isnan(val):
                                long_term_data[key].append(val)

                        try:
                            append_if_valid("RSI", indicators.get("rsi_14"))
                            append_if_valid("MACD", indicators.get("macd"))
                            append_if_valid("MACD_HIST", indicators.get("macd_hist"))
                            append_if_valid("EMA_DIST", indicators.get("price") - indicators.get("ema_20") if "price" in indicators and "ema_20" in indicators else None)
                            append_if_valid("ATR", indicators.get("atr_14"))
                            append_if_valid("ADX", indicators.get("adx_14"))
                            append_if_valid("VWAP", indicators.get("vwap"))
                            bb_width = (indicators.get("bb_upper") - indicators.get("bb_lower")) if indicators.get("bb_upper") and indicators.get("bb_lower") else None
                            append_if_valid("BB_WIDTH", bb_width)
                            append_if_valid("VIX", indicators.get("vix"))
                            append_if_valid("SPY_ABS", abs(indicators.get("price", 0)))

                            append_if_valid("IV", option_data.get("iv"))
                            append_if_valid("DELTA", option_data.get("delta"))
                            append_if_valid("SIZE", position_size)

                            for k in long_term_data:
                                if len(long_term_data[k]) > 500:
                                    long_term_data[k] = long_term_data[k][-500:]
                        except Exception:
                            logger.error(f"⚠️ Error updating long_term_data from trade {trade_idx + 1}")
                            traceback.print_exc()

                    else:
                        logger.debug(f"❌ Trade {trade_idx + 1} skipped or failed (simulate_trade returned None)")

                except Exception as trade_err:
                    logger.error(f"🔥 Exception in simulate_trade() for Day {day + 1}, Trade {trade_idx + 1}")
                    traceback.print_exc()

            # === Write meta log if trades exist ===
            if trades:
                try:
                    with open(META_LOG_PATH, "a") as f:
                        for t in trades:
                            f.write(json.dumps(t) + "\n")
                    logger.debug(f"📝 Day {day + 1}: Logged {len(trades)} trades to {META_LOG_PATH}")
                except Exception:
                    logger.error(f"⚠️ Failed to write trades for Day {day + 1} to {META_LOG_PATH}")
                    traceback.print_exc()
            else:
                logger.debug(f"🕳️ Day {day + 1}: No trades generated")

            logger.info(f"📊 Day {day + 1}: {successful_trades}/{TRADES_PER_DAY} trades returned from simulate_trade()")

            if (day + 1) % 50 == 0:
                logger.info(f"📆 Simulated {day + 1} days.")

        except Exception as day_err:
            logger.error(f"💥 Fatal error in simulation for Day {day + 1}")
            traceback.print_exc()

    logger.info("✅ Simulation complete.")

    try:
        skipped_trades = SIM_DAYS * TRADES_PER_DAY - len(TRADE_HISTORY)
        diagnostics = {
            "Meta actions made": meta_stats.get("decisions_made", 0),
            "Empty states": meta_stats.get("empty_states", 0),
            "Missing classifier": meta_stats.get("no_classifier", 0),
            "Other skips": meta_stats.get("other_skips", 0),
        }
        summary_msg = summarize_simulation_results(TRADE_HISTORY, skipped_trades, diagnostics)
        send_telegram_message(summary_msg)
    except Exception:
        logger.warning("⚠️ Failed to send Telegram simulation summary.")
        traceback.print_exc()


if __name__ == "__main__":
    main()