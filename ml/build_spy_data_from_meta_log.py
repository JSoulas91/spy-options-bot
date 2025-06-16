import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import calculate_indicators

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_NPZ = Path("ml/dataset.npz")
OUTPUT_CSV = Path("ml/spy_data.csv")

FEATURE_NAMES = [
    "pnl",
    "confidence",
    "setup_quality",
    "vix",
    "realized_vol",
    "trade_type",
    "total_signals_today",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "vwap",
    "atr_14",
    "adx_14"
]


def extract_features(entry: dict) -> tuple[np.ndarray, int, str] | None:
    trade = entry.get("trade", {})
    bar = entry.get("bar", {})
    market = entry.get("market", {})
    timestamp = entry.get("timestamp")

    open_, high, low, close = bar.get("open"), bar.get("high"), bar.get("low"), bar.get("close")
    volume = bar.get("volume", 1.0)

    if None in (open_, high, low, close):
        bot_logger.warning("[Feature Extract] Skipping entry due to missing OHLC data")
        return None

    try:
        df_bar = pd.DataFrame([{
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }])

        df_ind = calculate_indicators(df_bar)
        row = df_ind.iloc[0]

        # Helper to get value or fallback
        def safe_val(name: str, fallback: float = 0.0):
            return float(row[name]) if name in row and pd.notna(row[name]) else fallback

        pnl = float(trade.get("pnl", 0.0))
        confidence = float(trade.get("confidence", 0.0))
        setup_quality = float(trade.get("setup_quality", 0.5))
        vix = float(market.get("vix", 20.0))
        realized_vol = float(market.get("realized_vol", 0.02))
        trade_type = int(trade.get("trade_type", 0))
        total_signals_today = int(trade.get("total_signals_today", 0))

        # VWAP fallback to close if not present
        vwap_value = safe_val("VWAP", fallback=close)

        features = np.array([
            pnl,
            confidence,
            setup_quality,
            vix,
            realized_vol,
            trade_type,
            total_signals_today,
            safe_val("EMA_20"),
            safe_val("RSI_14", fallback=50.0),
            safe_val("MACD"),
            safe_val("MACD_signal"),
            safe_val("MACD_hist"),
            safe_val("BB_upper"),
            safe_val("BB_middle"),
            safe_val("BB_lower"),
            vwap_value,
            safe_val("ATR_14"),
            safe_val("ADX_14")
        ], dtype=np.float32)

        label = 1 if pnl > 5 else 0

        return features, label, timestamp

    except Exception as e:
        available_cols = list(df_ind.columns)
        bot_logger.warning(f"[Feature Extract] Skipping entry due to error: {e}. Available columns: {available_cols}")
        return None


def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Log file not found: {META_LOG_PATH}")
        return

    features, labels, timestamps = [], [], []

    with META_LOG_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                result = extract_features(entry)
                if result:
                    feat, label, ts = result
                    features.append(feat)
                    labels.append(label)
                    timestamps.append(ts)
            except json.JSONDecodeError:
                continue

    if len(features) == 0:
        bot_logger.error("[Build Dataset] No valid entries found.")
        return

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    np.savez_compressed(OUTPUT_NPZ, X=features, y=labels, timestamps=np.array(timestamps))

    df_out = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_out.insert(0, "timestamp", timestamps)
    df_out["label"] = labels
    df_out.to_csv(OUTPUT_CSV, index=False)

    bot_logger.info(f"[Build Dataset] ✅ Saved {len(features)} entries to {OUTPUT_NPZ} and {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()