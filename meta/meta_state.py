"""
Meta‑state builders:
• Dynamic feature normalization with long‑term auto‑scaling
• Live SPY & option quote for exit state
• Position‑size awareness for entry state
"""

from __future__ import annotations
import time, numpy as np, pytz
from datetime import datetime
from typing import Dict, Tuple, List

from utils.logger import bot_logger as logger
from utils.vix_utils import get_vix_level
from data.multi_timeframe_fetcher import get_minutes_since_open, get_spy_latest_quote
from data.options_fetcher import get_quote as get_option_quote
from config import MAX_POSITION_SIZE  # ← NEW

eastern = pytz.timezone("US/Eastern")

DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
    "RSI":(0,100),"MACD":(-5,5),"EMA_DIST":(-10,10),"VOL":(0,10_000_000),
    "CONF":(0,1),"DURATION":(0,390),"PROFIT":(-1,1),"VIX":(10,40),
    "SPY_ABS":(350,500),"IV":(0,1),"DELTA":(-1,1),
    "SIZE":(0, MAX_POSITION_SIZE),               # ← NEW
}

def normalize(val: float, rng: Tuple[float,float]) -> float:
    lo, hi = rng
    return float(max(0, min(1, (val-lo)/(hi-lo+1e-9)))) if hi-lo else 0.5

_DYNAMIC_CACHE: Dict[str, Tuple[Tuple[float,float], float]] = {}
_DYNAMIC_TTL = 3600

def _calc_range(feature: str, lt: Dict[str,np.ndarray]) -> Tuple[float,float]:
    vals: List[float] = []
    for df in lt.values():
        if df is None or df.empty: continue
        if feature=="EMA_DIST":
            vals.extend((df["price"]-df["ema_20"]).tolist())
        else:
            vals.extend(df.get(feature,[]).tolist())
    return (min(vals),max(vals)) if vals else DEFAULT_RANGES[feature]

def get_range(feature:str, lt)->Tuple[float,float]:
    now=time.time()
    if feature in _DYNAMIC_CACHE and now-_DYNAMIC_CACHE[feature][1]<_DYNAMIC_TTL:
        return _DYNAMIC_CACHE[feature][0]
    rng=_calc_range(feature,lt)
    if rng[0]==rng[1]: rng=DEFAULT_RANGES[feature]
    _DYNAMIC_CACHE[feature]=(rng,now)
    return rng

def summarize_past_trades(trades, rng_p, rng_d):
    if not trades: return [0.5,0.5]
    prof=[t.get("profit",0) for t in trades]
    dur =[t.get("duration",0) for t in trades]
    return [normalize(np.mean(prof),rng_p), normalize(np.mean(dur),rng_d)]

# ───────────────────────── ENTRY BUILDER
def build_meta_state_for_entry(
    data_1m, data_5m, data_15m, data_1h, data_1d,
    confidence_score: float, trade_type: int,
    past_trades=None,long_term_data=None,
    position_size: float = 0.0                # ← NEW
)->np.ndarray:
    past_trades   = past_trades or []
    long_term_data= long_term_data or {}

    try:
        rsi_rng=get_range("RSI",long_term_data)
        macd_rng=get_range("MACD",long_term_data)
        ema_rng=get_range("EMA_DIST",long_term_data)
        vol_rng=get_range("VOL",long_term_data)
        dur_rng=DEFAULT_RANGES["DURATION"]
        prof_rng=DEFAULT_RANGES["PROFIT"]

        def tf(d):
            last=d.iloc[-1]
            return [
                normalize(last.get("rsi",50),rsi_rng),
                normalize(last.get("macd",0),macd_rng),
                normalize(last.get("price",0)-last.get("ema_20",0),ema_rng),
                normalize(last.get("volume",0),vol_rng),
            ]
        state=[
            normalize(confidence_score,DEFAULT_RANGES["CONF"]),
            1.0 if trade_type==1 else 0.0,
            normalize(get_minutes_since_open(),dur_rng),
            normalize(get_vix_level(),DEFAULT_RANGES["VIX"]),
            # position‑size awareness
            normalize(position_size,DEFAULT_RANGES["SIZE"]),
            normalize(position_size*get_vix_level(),(0,MAX_POSITION_SIZE*40)),
            *summarize_past_trades(past_trades,prof_rng,dur_rng),
            *tf(data_1m),*tf(data_5m),*tf(data_15m),*tf(data_1h),*tf(data_1d),
        ]
        # long‑term trends
        for p in ["5d","10d","15d","1mo","3mo","6mo"]:
            df=long_term_data.get(p)
            if df is not None and not df.empty:
                last=df.iloc[-1]
                state+=[
                    normalize(last.get("rsi",50),rsi_rng),
                    normalize(last.get("macd",0),macd_rng),
                    normalize(last.get("price",0)-last.get("ema_20",0),ema_rng),
                ]
            else:
                state+=[0.5,0.5,0.5]
        return np.array(state,dtype=np.float32)
    except Exception as e:
        logger.error(f"[MetaState] entry error: {e}")
        return np.zeros(70,dtype=np.float32)  # length stable

# ───────────────────────── EXIT CACHES / BUILDER (unchanged except ranges remain)
_OPTION_CACHE: Dict[str, Tuple[dict,float]] = {}
def _cached_option_quote(sym:str,ttl=6):
    now=time.time()
    if sym in _OPTION_CACHE and now-_OPTION_CACHE[sym][1]<ttl:
        return _OPTION_CACHE[sym][0]
    q=get_option_quote(sym); _OPTION_CACHE[sym]=(q,now)
    return q

def build_meta_state_for_exit(trade:dict,past_trades=None,long_term_data=None)->np.ndarray:
    past_trades=past_trades or []; long_term_data=long_term_data or {}
    try:
        spy_q=get_spy_latest_quote() or {}; spy_price=spy_q.get("price",0)
        opt_sym=trade.get("option_symbol"); opt_q=_cached_option_quote(opt_sym) if opt_sym else {}
        iv=float(opt_q.get("iv",0) or 0); delta=float(opt_q.get("delta",0) or 0)

        dur_rng=DEFAULT_RANGES["DURATION"]; prof_rng=DEFAULT_RANGES["PROFIT"]
        spy_abs_rng=DEFAULT_RANGES["SPY_ABS"]

        confidence=trade.get("confidence",0.5)
        trade_type=trade.get("trade_type",0)
        entry_price=trade.get("entry_price",0)
        pnl_pct=(spy_price-entry_price)/max(entry_price,1e-9)
        minutes_open=normalize(
            (datetime.now(eastern)-datetime.fromisoformat(trade.get("timestamp")).astimezone(eastern)).seconds//60,
            dur_rng)

        state=[
            normalize(confidence,DEFAULT_RANGES["CONF"]),
            1.0 if trade_type==1 else 0.0,
            normalize(get_minutes_since_open(),dur_rng),
            minutes_open,
            normalize(pnl_pct,prof_rng),
            normalize(get_vix_level(),DEFAULT_RANGES["VIX"]),
            *summarize_past_trades(past_trades,prof_rng,dur_rng),
            normalize(spy_price-entry_price,DEFAULT_RANGES["EMA_DIST"]),
            normalize(spy_price,spy_abs_rng),
            normalize(iv,DEFAULT_RANGES["IV"]),
            normalize(delta,DEFAULT_RANGES["DELTA"]),
        ]
        return np.array(state,dtype=np.float32)
    except Exception as e:
        logger.error(f"[MetaState] exit error: {e}")
        return np.zeros(40,dtype=np.float32)