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

from typing import List, Dict, Optional
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
SIM_DAYS = 500
TRADES_PER_DAY = 12
GBM_MU = 0.08
GBM_SIGMA = 0.22
START_PRICE = 450.0
WARM_UP_DAYS = 110
PAD_VAL = 0.5
STATE_SEQUENCE_LENGTH = 20  # Adjust if your model uses more or fewer timesteps
STATE_DIM = 83             # Must match what your model expects per timestep

ACCUMULATED_CLOSES = []
ACCUMULATED_VOLUMES = []
TRADE_HISTORY = []
LONG_TERM_DATA = {
    "RSI": [],
    "MACD": [],
    "MACD_HIST": [],
    "EMA_DIST": [],
    "ATR": [],
    "ADX": [],
    "VWAP": [],
    "BB_WIDTH": [],
    "VIX": [],
    "SPY_ABS": [],
    "IV": [],
    "DELTA": [],
    "SIZE": [],
    "DURATION": [],
    "PROFIT": [],
}

META_LOG_PATH = Path("meta/meta_log.jsonl")
RNG = random.Random(42)
GARBAGE_KEEP_PROB = 0.05
COMMISSION_PER_CONTRACT = 0.35
CONTRACT_MULTIPLIER = 100  # Options multiplier

meta_agent = MetaAgent()
model_inference = ModelInference()

print(f"SIM_DAYS={SIM_DAYS}, TRADES_PER_DAY={TRADES_PER_DAY}, START_PRICE={START_PRICE}")

def is_padded(meta):
    meta = np.array(meta)
    return np.allclose(meta, 0.5, atol=1e-6)

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

def normalize(value, value_range):
    min_val, max_val = value_range
    if max_val == min_val:
        return PAD_VAL
    return max(PAD_VAL, min(1.0, (value - min_val) / (max_val - min_val)))

def get_range(feat: str, long_term) -> Tuple[float, float]:
    now = time.time()
    if feat in _DYNAMIC and now - _DYNAMIC[feat][1] < _DYN_TTL:
        return _DYNAMIC[feat][0]
    rng = _calc_range(feat, long_term)
    if rng[0] == rng[1]:
        rng = DEFAULT_RANGES[feat]
    _DYNAMIC[feat] = (rng, now)
    return rng
    
def update_long_term_stats(long_term_data: dict, features: dict):
    for key, val in features.items():
        if key not in long_term_data:
            long_term_data[key] = []
        long_term_data[key].append(val)
        # Limit history size to avoid memory issues
        if len(long_term_data[key]) > 5000:
            long_term_data[key] = long_term_data[key][-5000:]

def _pad(vec: List[float], dim: int = STATE_DIM) -> List[float]:
    if len(vec) < dim:
        return vec + [PAD_VAL] * (dim - len(vec))
    return vec[:dim]
    
def _regime_one_hot(regime: str) -> List[float]:
    mapping = {"bull": [1.0, 0.0, 0.0], "bear": [0.0, 1.0, 0.0], "sideways": [0.0, 0.0, 1.0]}
    return mapping.get(regime, [PAD_VAL, PAD_VAL, PAD_VAL])
    
def _classify_regime(day_bar, vix_val: float) -> str:
    if vix_val > 30 or day_bar["rsi"] < 40:
        return "bear"
    elif vix_val < 20 and day_bar["rsi"] > 60:
        return "bull"
    else:
        return "sideways"
        
def fetch_vix_price() -> Optional[float]:
    try:
        # REPLACE with real VIX fetching if available
        return 18.0
    except Exception:
        return None
        
def get_minutes_since_open() -> int:
    from datetime import datetime
    now = datetime.utcnow()
    market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 AM ET = 13:30 UTC
    delta = now - market_open
    return max(0, delta.seconds // 60)
    
def summarise_past(past_trades: List[dict], profit_range: tuple, dur_range: tuple) -> List[float]:
    if not past_trades:
        return [PAD_VAL] * 6

    profits = [t.get("pnl", 0.0) for t in past_trades[-3:]]
    durations = [t.get("duration", 0.0) for t in past_trades[-3:]]

    normalized_profits = [normalize(p, profit_range) for p in profits]
    normalized_durations = [normalize(d, dur_range) for d in durations]

    return normalized_profits + normalized_durations
    
def make_option_symbol(day: datetime, strike: float, c_or_p: str) -> str:
    symbol = f"SPY{day.strftime('%y%m%d')}{c_or_p}{int(strike*100):08d}"
    print(f"Option symbol created: {symbol}")
    return symbol


def construct_bars(prices, volumes, interval, start_time=None):
    if start_time is None:
        start_time = datetime(2025, 1, 1, 9, 30)

    if len(prices) != len(volumes):
        logger.debug(f"⚠️ construct_bars: length mismatch - prices={len(prices)}, volumes={len(volumes)}")
        return []

    if len(prices) < interval:
        logger.debug(f"⚠️ construct_bars: not enough data to form one bar - len(prices)={len(prices)}, interval={interval}")
        return []

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
            "open": chunk[0],
            "high": max(chunk),
            "low": min(chunk),
            "close": chunk[-1],
            "volume": sum(vol_chunk)
        }

        bars.append(bar)

    logger.debug(f"✅ construct_bars: constructed {len(bars)} bars at interval={interval}min starting from {start_time.strftime('%Y-%m-%d %H:%M')}")
    return bars


def compute_all_indicators(prices, volumes, idx):
    start_idx = max(0, idx - 100)
    window = prices[start_idx:idx + 1]
    vol_window = volumes[start_idx:idx + 1]

    # --- [TRACE] Raw input debug ---
    print(f"[TRACE] window raw (len={len(window)}): {window}")
    print(f"[TRACE] vol_window raw (len={len(vol_window)}): {vol_window}")

    # Ensure numeric and clean closes
    closes_series = pd.to_numeric(pd.Series(window), errors="coerce")
    vol_series = pd.to_numeric(pd.Series(vol_window), errors="coerce").fillna(0.0)

    # --- [TRACE] Parsed debug ---
    print(f"[TRACE] closes_series (len={len(closes_series)}): {closes_series.tolist()}")
    print(f"[TRACE] vol_series (len={len(vol_series)}): {vol_series.tolist()}")

    # Align volume with valid prices only
    valid_mask = closes_series.notna()

    if valid_mask.sum() == 0:
        print(f"[ERROR] All closes are NaN at idx={idx}, cannot compute indicators")
        return None

    # --- [DEBUG] NaN filtering summary ---
    print(f"[DEBUG] Raw window (prices): {window[-5:]}")
    print(f"[DEBUG] Parsed closes_series: {closes_series[-5:]}")
    print(f"[DEBUG] Valid closes count: {valid_mask.sum()}")

    closes = closes_series[valid_mask]
    vol_series = vol_series[valid_mask]
    window = closes.values  # clean numeric window used for VWAP

    if len(closes) < 20:
        print(f"[DEBUG] idx={idx}, closes length: {len(closes)}, vol_series length: {len(vol_series)}")
        print(f"[DEBUG] closes (last 5):\n{closes.tail()}")
        print(f"[DEBUG] vol_series (last 5):\n{vol_series.tail()}")
        print(f"[WARNING] Not enough valid numeric data for indicators at idx={idx}")
        return None

    indicators = {}
    print(f"[DEBUG] idx={idx}, using window from {start_idx} to {idx} (length={len(closes)})")
    
    # EMA 20
    ema_20 = closes.ewm(span=20).mean().iloc[-1]
    if pd.isna(ema_20):
        print(f"[WARNING] EMA 20 is NaN at idx={idx}")
    indicators["ema_20"] = ema_20

    # RSI 14
    delta = closes.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean().iloc[-1]
    avg_loss = down.rolling(window=14).mean().iloc[-1]

    if avg_loss == 0:
        print(f"[WARNING] avg_loss is zero at idx={idx}, adjusting to avoid div by zero")
        avg_loss = 1e-6

    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    if pd.isna(rsi_14):
        print(f"[WARNING] RSI 14 is NaN at idx={idx}")
    indicators["rsi_14"] = rsi_14

    # MACD
    exp1 = closes.ewm(span=12, adjust=False).mean()
    exp2 = closes.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    if pd.isna(macd.iloc[-1]) or pd.isna(signal.iloc[-1]):
        print(f"[WARNING] MACD or signal line is NaN at idx={idx}")

    indicators["macd"] = macd.iloc[-1]
    indicators["macd_signal"] = signal.iloc[-1]
    indicators["macd_hist"] = (macd - signal).iloc[-1]

    # Bollinger Bands
    std = closes.rolling(window=20).std().iloc[-1]
    middle = closes.rolling(window=20).mean().iloc[-1]
    if pd.isna(std) or pd.isna(middle):
        print(f"[WARNING] Bollinger Bands std or middle is NaN at idx={idx}")

    indicators["bb_middle"] = middle
    indicators["bb_upper"] = middle + 2 * std
    indicators["bb_lower"] = middle - 2 * std

    # VWAP
    if len(window) != len(vol_series):
        print(f"[ERROR] VWAP length mismatch: prices={len(window)}, volumes={len(vol_series)} at idx={idx}")
        return None
    else:
        vwap = np.average(window, weights=vol_series)
        indicators["vwap"] = vwap

    # ATR 14
    tr_list = []
    for i in range(1, len(closes)):
        tr_value = max(closes.iloc[i] - closes.iloc[i - 1], 0)
        tr_list.append(tr_value)
    tr = pd.Series(tr_list)
    atr_14 = tr.rolling(window=14).mean().iloc[-1]
    if pd.isna(atr_14):
        print(f"[WARNING] ATR 14 is NaN at idx={idx}")
    indicators["atr_14"] = atr_14

    # ADX (simulated)
    adx = RNG.uniform(10, 35)
    indicators["adx_14"] = adx
    
    indicators["price"] = closes[-1] if closes else None

    # Round and print final indicators
    for k in indicators:
        try:
            indicators[k] = round(float(indicators[k]), 4)
        except Exception as e:
            print(f"[ERROR] Rounding indicator '{k}' failed at idx={idx} with error: {e}")
            indicators[k] = None

    print(f"[INFO] Indicators at idx={idx}: {indicators}")
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
        
def black_scholes_delta(s, k, t, r, sigma, call=True):
    if t <= 0:
        return 1.0 if call and s > k else 0.0 if call else -1.0 if s < k else 0.0
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return nd1 if call else nd1 - 1

def clean_bars(bars):
    return [bar for bar in bars if all(isinstance(bar.get(k), (int, float)) for k in ["open", "high", "low", "close", "volume"])]
    

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

    try:
        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            if df is None:
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
            return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]

        vix_val = fetch_vix_price() or 20.0
        regime = classifier_output.get("regime_class") if classifier_output and "regime_class" in classifier_output else _classify_regime(data_1d.iloc[-1], vix_val)

        clf_conf = classifier_output.get("trade_success_prob") if classifier_output else None
        norm_conf = normalize(clf_conf if clf_conf is not None else confidence_score, DEFAULT_RANGES["CONF"])

        state: List[float] = [
            norm_conf,
            1.0 if trade_type == 1 else 0.0,
            normalize(get_minutes_since_open(), dur_rng),
            normalize(vix_val, DEFAULT_RANGES["VIX"]),
            normalize(position_size, DEFAULT_RANGES["SIZE"]),
            *_regime_one_hot(regime),
            *summarise_past(past_trades, prof_rng, dur_rng),
            *tf_feats(data_1m), *tf_feats(data_5m),
            *tf_feats(data_15m), *tf_feats(data_1h), *tf_feats(data_1d),
        ]

        for p in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            df = long_term_data.get(p)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                state += [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                ]
            else:
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
            if len(class_probs) != 3:
                class_probs = [PAD_VAL] * 3
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            state += [PAD_VAL] * 8

        return build_sequence(state)

    except Exception as e:
        logger.error(f"Error building meta state for entry: {e}")
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

    try:
        rsi_rng = get_range("RSI", long_term_data)
        macd_rng = get_range("MACD", long_term_data)
        ema_rng = get_range("EMA_DIST", long_term_data)
        vol_rng = get_range("VOL", long_term_data)
        dur_rng = DEFAULT_RANGES["DURATION"]
        prof_rng = DEFAULT_RANGES["PROFIT"]

        def tf_feats(df):
            if df is None:
                return [normalize(50, rsi_rng), normalize(0, macd_rng), normalize(0, ema_rng), normalize(0, vol_rng)]
            if hasattr(df, "iloc") and len(df) > 0:
                last = df.iloc[-1]
                return [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                    normalize(last.get("volume", 0), vol_rng),
                ]
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
            *summarise_past(past_trades, prof_rng, dur_rng),
            norm_pnl,
            *tf_feats(data_1m), *tf_feats(data_5m),
            *tf_feats(data_15m), *tf_feats(data_1h), *tf_feats(data_1d),
        ]

        for p in ["5d", "10d", "15d", "1mo", "3mo", "6mo"]:
            df = long_term_data.get(p)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                state += [
                    normalize(last.get("rsi", 50), rsi_rng),
                    normalize(last.get("macd", 0), macd_rng),
                    normalize(last.get("price", 0) - last.get("ema_20", 0), ema_rng),
                ]
            else:
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
            if len(class_probs) != 3:
                class_probs = [PAD_VAL] * 3
            state.extend(class_probs)

            entropy = classifier_output.get("entropy", PAD_VAL)
            state.append(entropy if 0 <= entropy <= 1 else PAD_VAL)
        else:
            state += [PAD_VAL] * 8

        return build_sequence(state)

    except Exception as e:
        logger.error(f"Error building meta state for exit: {e}")
        return np.stack([_pad([PAD_VAL] * STATE_DIM) for _ in range(STATE_SEQUENCE_LENGTH)], axis=0)


def simulate_trade(day, trade_idx, closes, volumes, vix_shift):
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
    
    # Extra buffer bars to ensure indicators can initialize
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
    
    # Compute earliest bar required among all timeframes
    min_required_minutes = max(input_slices.values())
    if trade_minute < min_required_minutes:
        logger.debug(
            f"⏩ Skipping trade {trade_idx} on day {day}: "
            f"trade_minute={trade_minute} < min_required_minutes={min_required_minutes}"
        )
        return None

    total_offset = timedelta(days=day, minutes=trade_minute)
    base_time = datetime(2025, 1, 1, 9, 30) + total_offset

    try:
        logger.debug(f"🔧 Constructing bars using closes[{trade_minute - max_required}:{trade_minute}]")

        bars_1m = construct_bars(
            closes[trade_minute - input_slices["1m"]:trade_minute],
            volumes[trade_minute - input_slices["1m"]:trade_minute],
            1,
            start_time=base_time - timedelta(minutes=input_slices["1m"] - 1)
        )
        
        bars_5m = construct_bars(
            closes[trade_minute - input_slices["5m"]:trade_minute],
            volumes[trade_minute - input_slices["5m"]:trade_minute],
            5,
            start_time=base_time - timedelta(minutes=input_slices["5m"] - 5)
        )
        
        bars_15m = construct_bars(
            closes[trade_minute - input_slices["15m"]:trade_minute],
            volumes[trade_minute - input_slices["15m"]:trade_minute],
            15,
            start_time=base_time - timedelta(minutes=input_slices["15m"] - 15)
        )
        
        bars_1h = construct_bars(
            closes[trade_minute - input_slices["1h"]:trade_minute],
            volumes[trade_minute - input_slices["1h"]:trade_minute],
            60,
            start_time=base_time - timedelta(minutes=input_slices["1h"] - 60)
        )
        
        bars_1d = construct_bars(
            closes[trade_minute - input_slices["1d"]:trade_minute],
            volumes[trade_minute - input_slices["1d"]:trade_minute],
            390,
            start_time=base_time - timedelta(minutes=input_slices["1d"] - 390)
        )
    except Exception as e:
        logger.exception(f"❌ Exception during bar construction: {e}")
        return None

    # Clean bars
    bars_1m = clean_bars(bars_1m)
    bars_5m = clean_bars(bars_5m)
    bars_15m = clean_bars(bars_15m)
    bars_1h = clean_bars(bars_1h)
    bars_1d = clean_bars(bars_1d)

    # Validate lengths
    required_bars = required_lookback
    for tf, bars in zip(required_bars, [bars_1m, bars_5m, bars_15m, bars_1h, bars_1d]):
        if not isinstance(bars, list) or len(bars) < required_bars[tf]:
            logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: only {len(bars)} bars for {tf} (required: {required_bars[tf]})")
            return None
        logger.debug(f"✅ {tf} bars OK: {len(bars)}")
        
    #ohlcv construction
    closes_1m = [bar["close"] for bar in bars_1m]
    opens_1m = [bar["open"] for bar in bars_1m]
    highs_1m = [bar["high"] for bar in bars_1m]
    lows_1m = [bar["low"] for bar in bars_1m]
    volumes_1m = [bar["volume"] for bar in bars_1m]
    
    logger.debug(f"📊 Extracted OHLCV arrays from 1m bars:")
    logger.debug(f"   • closes_1m[0:3]: {closes_1m[:3]} ... len={len(closes_1m)}")
    logger.debug(f"   • opens_1m[0:3]: {opens_1m[:3]} ... len={len(opens_1m)}")
    logger.debug(f"   • highs_1m[0:3]: {highs_1m[:3]} ... len={len(highs_1m)}")
    logger.debug(f"   • lows_1m[0:3]: {lows_1m[:3]} ... len={len(lows_1m)}")
    logger.debug(f"   • volumes_1m[0:3]: {volumes_1m[:3]} ... len={len(volumes_1m)}")

    if len(bars_1m) < required_bars["1m"]:
        logger.debug(
            f"⏩ Skipping trade {trade_idx} on day {day}: "
            f"len(bars_1m) ({len(bars_1m)}) < required_bars['1m'] ({required_bars['1m']}) - insufficient bars to simulate trade entry."
        )
        return None

    max_start_idx = len(bars_1m) - required_bars["1m"]
    logger.debug(
        f"Computed max_start_idx = {max_start_idx} (len(bars_1m)={len(bars_1m)} - required_bars['1m']={required_bars['1m']})"
    )

    if max_start_idx < 0:
        logger.debug(
            f"⏩ Skipping trade {trade_idx} on day {day}: "
            f"max_start_idx ({max_start_idx}) < 0 - not enough room to select a valid start index."
        )
        return None

    start_idx = RNG.randint(0, max_start_idx)
    logger.debug(f"Selected start_idx={start_idx} within [0, {max_start_idx}]")
    
    price_sig = closes_1m[start_idx]
    logger.debug(f"Price signal at start_idx={start_idx} is {price_sig}")
    
    strike = round(price_sig + RNG.uniform(-6, 6), 1)
    logger.debug(f"Generated strike price: {strike}")
    
    option_type = RNG.choice(["C", "P"])
    logger.debug(f"Selected option type: {option_type}")
    
    expiry_days = RNG.randint(7, 30)
    t_expiry = expiry_days / 365
    logger.debug(f"Option expiry_days={expiry_days}, t_expiry={t_expiry:.4f} years")
    
    day_dt = base_time + timedelta(days=day)
    option_symbol = make_option_symbol(day_dt, strike, option_type)
    
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
            "iv": 0.25,  # constant implied volatility for now
            "delta": option_delta
        }
    
        logger.debug(f"Computed Black-Scholes price: {option_price:.4f}, delta: {option_delta:.4f}")
    
    except Exception as e:
        logger.error(f"Error computing Black-Scholes price/delta: {e}")
        option_data = {
            "price": 0.0,
            "iv": 0.25,
            "delta": 0.0
        }
    
    slippage = RNG.uniform(-0.5, 0.5) / 100
    fill_pct = RNG.uniform(0.7, 1.0)
    entry_price = round(option_price * (1 + slippage), 2)
    logger.debug(f"Simulated slippage: {slippage*100:.2f}%, fill_pct: {fill_pct:.3f}, entry_price after slippage: {entry_price}")
    
    is_swing = RNG.random() < 0.2  # random swing trade decision
    logger.debug(f"is_swing trade decision: {is_swing}")
    
    if len(bars_1m) < 101:
        logger.debug("Not enough data for indicators, skipping trade simulation")
        return None


    try:
        indicators = compute_all_indicators(closes_1m, volumes_1m, len(closes_1m) - 1)
        if not indicators or any(
            v is None or (isinstance(v, float) and not (v == v))  # NaN check
            for v in indicators.values()
        ):
            logger.warning(f"Invalid or incomplete indicators at idx={len(bars_1m) - 1}")
            return None
        logger.debug(f"Indicators at {len(bars_1m)-1}: {indicators}")
    except Exception as e:
        logger.exception("Exception during indicator computation")
        return None
    
    # Classifier features
    classifier_confidence = round(np.random.beta(5, 2), 2)
    logger.debug(f"Simulated classifier confidence: {classifier_confidence}")
    
    setup_quality = round(RNG.uniform(0.6, 1.0), 2)
    logger.debug(f"Simulated setup quality: {setup_quality}")
    
    vix = round(RNG.uniform(15, 35), 2)
    logger.debug(f"Simulated VIX: {vix}")
    
    if start_idx >= 20:
        realized_vol = round(np.std(closes_1m[start_idx - 20:start_idx]), 2)
        logger.debug(f"Calculated realized_vol over last 20 prices: {realized_vol}")
    else:
        realized_vol = 1.5
        logger.debug(f"Insufficient data for realized_vol, defaulting to: {realized_vol}")
    
    trade_type = 0 if option_type == "C" else 1
    logger.debug(f"Encoded trade_type: {trade_type} (0=Call, 1=Put)")
    
    total_signals_today = RNG.randint(0, 10)
    logger.debug(f"Simulated total_signals_today: {total_signals_today}")

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

    # Grab past N trades for meta-state summary
    past_trades = TRADE_HISTORY[-10:] if len(TRADE_HISTORY) >= 10 else TRADE_HISTORY
    
    # Build feature vector
    features_df = build_features_for_trade(classifier_features)
    logger.debug(f"🛠️ Built features DataFrame: type={type(features_df)}, shape={getattr(features_df, 'shape', 'N/A')}")

    update_long_term_stats(long_term_data, classifier_features)
    
    # Ensure it's a proper DataFrame
    if not isinstance(features_df, pd.DataFrame):
        logger.debug("⚠️ build_features_for_trade returned non-DataFrame, coercing to DataFrame")
        features_df = pd.DataFrame([classifier_features])

    # Validate expected shape
    if features_df.shape[0] != 1:
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: features_df has invalid shape {features_df.shape}, expected (1, N)")
        return None

    try:
        inference = ModelInference()
        raw_output = inference.predict_with_confidence(features_df)
        classifier_output = ModelInference.wrap_classifier_output(raw_output)
    except Exception as e:
        logger.debug(f"Classifier prediction failed: {e}")
        return None
        
    # Dynamic position sizing based on classifier confidence
    confidence = classifier_output.get("trade_success_prob", 0.5)
    position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * confidence
    
    meta_entry = build_meta_state_for_entry(
        data_1m=bars_1m,
        data_5m=bars_5m,
        data_15m=bars_15m,
        data_1h=bars_1h,
        data_1d=bars_1d,
        position_size=position_size,  # you can set this to 1.0 if you don't have it dynamic
        confidence_score=classifier_confidence,
        trade_type=int(is_swing),
        past_trades=past_trades,
        long_term_data=long_term_data,
        classifier_output=classifier_output
    )
    
    if meta_entry is None or is_padded(meta_entry):
        logger.debug(f"🚫 Skipping trade {trade_idx} on day {day}: entry_meta is invalid or padded")
        return None
    
    action, agent_confidence = meta_agent.select_action(meta_entry)
    logger.debug(f"🎯 Trade {trade_idx} on day {day}: Meta-agent selected action {action} with confidence {agent_confidence:.2f}")

    if action == 0:
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: Meta-agent chose to skip (action=0)")
        return None

    duration = RNG.randint(10, 40) if not is_swing else RNG.randint(100, 300)
    logger.debug(f"📈 Trade {trade_idx} on day {day}: Duration={duration}, start_idx={start_idx}, total_prices={len(closes_1m)}")

    if start_idx + duration >= len(closes_1m):
        logger.debug(f"⏩ Skipping trade {trade_idx} on day {day}: Not enough future bars for trade duration ({start_idx + duration} >= {len(closes_1m)})")
        return None

    final_price = closes_1m[start_idx + duration]

    minutes_per_year = 252 * 6.5 * 60
    time_left = max(t_expiry - (duration * 1) / minutes_per_year, 0.01)
    logger.debug(f"⏳ Time left to expiry: {time_left:.4f} years")

    new_option_price = black_scholes_price(
        s=final_price,
        k=strike,
        t=time_left,
        r=0.01,
        sigma=0.25,
        call=(option_type == "C")
    )
    exit_price = round(new_option_price * (1 + slippage), 2)
    logger.debug(f"💸 Exit option price (slippage-adjusted): {exit_price:.2f} | Final SPY: {final_price:.2f}")

    gross_pnl = (exit_price - entry_price) * CONTRACT_MULTIPLIER * fill_pct
    total_commission = 2 * COMMISSION_PER_CONTRACT
    raw_pnl = gross_pnl - total_commission
    initial_cost = entry_price * CONTRACT_MULTIPLIER * fill_pct + 1e-9
    pct_pnl = (raw_pnl / initial_cost) * 100
    trade_result = pct_pnl
    logger.debug(f"📊 PnL: Gross={gross_pnl:.2f}, Raw={raw_pnl:.2f}, Pct={pct_pnl:.2f}%")
    
    atr = indicators.get('atr_14', 1.0)
    logger.debug(f"📐 ATR(14): {atr:.2f}")
    
    features_df = build_features_for_trade(classifier_features)
    try:
        inference = ModelInference()
        raw_output = inference.predict_with_confidence(features_df)
        classifier_output = ModelInference.wrap_classifier_output(raw_output)
    except Exception as e:
        logger.debug(f"Classifier prediction failed: {e}")
        return None
        
    # Dynamic position sizing based on classifier confidence
    confidence = classifier_output.get("trade_success_prob", 0.5)
    position_size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * confidence
    
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

    if meta_exit is None or is_padded(meta_exit):
        logger.debug(f"🚫 Skipping trade {trade_idx} on day {day}: exit_meta is invalid or padded")
        return None
        
    TRADE_HISTORY.append({
        "profit": trade_return,  # percentage PnL, e.g., 0.12 for +12%
        "duration": time_held_minutes,
        "position_size": position_size,
    })
    
    direction_correct = (
        (predicted_direction == 1 and final_price > price_sig) or
        (predicted_direction == 0 and final_price < price_sig)
    )
    logger.debug(f"🎯 Direction predicted: {predicted_direction}, Correct: {direction_correct}")
    
    shaped_reward = reward_shaper.compute_shaped_reward(
        trade_result={
            "pct_pnl": trade_result,
            "setup_quality": setup_quality,
            "entry_quality": abs(trade_result) / atr,
            "direction_correct": direction_correct,
            "trades_today": trade_idx,
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
    logger.debug(f"🏅 Shaped reward: {shaped_reward:.2f}")
    
    if shaped_reward < -2 and RNG.random() > GARBAGE_KEEP_PROB:
        logger.debug(f"🗑️ Skipping trade {trade_idx} on day {day}: shaped reward too low ({shaped_reward:.2f})")
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
    
    logger.debug(f"✅ Logged trade {trade_idx} on day {day} | PnL: {trade_result:.2f}% | Option: {option_symbol} | Duration: {duration}m")
    
    return {
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
        # ✅ New additions for warm-up data accumulation
        "indicators": indicators,
        "option_data": option_data,
        "position_size": position_size,
    }


def main():
    global ACCUMULATED_CLOSES, ACCUMULATED_VOLUMES

    for day in range(SIM_DAYS):
        # Warm-up period for multi-timeframe indicators
        if day < WARM_UP_DAYS:
            logger.debug(f"⏩ Skipping Day {day + 1}: Warming up multi-timeframe history")
            daily_prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
            daily_volumes = [RNG.randint(300_000, 1_000_000) for _ in daily_prices]
            ACCUMULATED_CLOSES.extend(daily_prices)
            ACCUMULATED_VOLUMES.extend(daily_volumes)
            continue

        vix_shift = RNG.uniform(14, 28)
        if RNG.random() < 0.08:
            vix_shift += RNG.uniform(5, 15)

        daily_prices = gbm_path(5000, START_PRICE, GBM_MU, GBM_SIGMA, 1 / 390)
        daily_volumes = [RNG.randint(300_000, 1_000_000) for _ in daily_prices]
        ACCUMULATED_CLOSES.extend(daily_prices)
        ACCUMULATED_VOLUMES.extend(daily_volumes)

        trades = []
        successful_trades = 0
        logger.debug(f"Day {day + 1}: Starting simulation with VIX shift {vix_shift:.2f}")

        for trade_idx in range(TRADES_PER_DAY):
            log_entry = simulate_trade(day, trade_idx, ACCUMULATED_CLOSES, ACCUMULATED_VOLUMES, vix_shift)

            if log_entry:
                trades.append(log_entry)
                successful_trades += 1
                logger.debug(f"✅ Trade {trade_idx + 1} generated: PnL={log_entry['pct_pnl']}, duration={log_entry['duration']}")

                # Accumulate long-term feature normalization stats during warm-up
                if day < WARM_UP_DAYS:
                    indicators = log_entry.get("indicators", {})
                    option_data = log_entry.get("option_data", {})
                    position_size = log_entry.get("position_size", 1.0)

                    def append_if_valid(key, val):
                        if key in LONG_TERM_DATA and isinstance(val, (int, float)) and not math.isnan(val):
                            LONG_TERM_DATA[key].append(val)

                    append_if_valid("RSI", indicators.get("rsi_14"))
                    append_if_valid("MACD", indicators.get("macd"))
                    append_if_valid("MACD_HIST", indicators.get("macd_hist"))
                    append_if_valid("EMA_DIST", indicators.get("price") - indicators.get("ema_20") if "price" in indicators and "ema_20" in indicators else 0)
                    append_if_valid("ATR", indicators.get("atr_14"))
                    append_if_valid("ADX", indicators.get("adx_14"))
                    append_if_valid("VWAP", indicators.get("vwap"))
                    append_if_valid("BB_WIDTH", (indicators.get("bb_upper") - indicators.get("bb_lower")) if indicators.get("bb_upper") and indicators.get("bb_lower") else None)
                    append_if_valid("VIX", indicators.get("vix"))
                    append_if_valid("SPY_ABS", abs(indicators.get("price", 0)))

                    append_if_valid("IV", option_data.get("iv"))
                    append_if_valid("DELTA", option_data.get("delta"))
                    append_if_valid("SIZE", position_size)

                    for k in LONG_TERM_DATA:
                        if len(LONG_TERM_DATA[k]) > 500:
                            LONG_TERM_DATA[k] = LONG_TERM_DATA[k][-500:]

            else:
                logger.debug(f"❌ Trade {trade_idx + 1} skipped or failed (simulate_trade returned None)")

        if trades:
            with open(META_LOG_PATH, "a") as f:
                for t in trades:
                    f.write(json.dumps(t) + "\n")
            logger.debug(f"Day {day + 1}: Logged {len(trades)} trades to {META_LOG_PATH}")
        else:
            logger.debug(f"Day {day + 1}: No trades generated")

        logger.info(f"Day {day + 1}: {successful_trades}/{TRADES_PER_DAY} trades returned from simulate_trade()")

        if (day + 1) % 50 == 0:
            logger.info(f"Simulated {day + 1} days.")

    logger.info("✅ Simulation complete.")
    send_telegram_message("✅ Simulation finished and saved to meta/meta_log.jsonl")


if __name__ == "__main__":
    main()