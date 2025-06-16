import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import calculate_indicators

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_NPZ = Path("ml/dataset.npz")
OUTPUT_CSV = Path("ml/spy_data.csv")

# Semantic feature column names
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

    open_ = bar.get("open")
    high = bar.get("high")
    low = bar.get("low")
    close = bar.get("close")
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

        # Compute indicators
        df_ind = calculate_indicators(df_bar)
        row = df_ind.iloc[0]

        # Primary features from logs
        pnl = float(trade.get("pnl", 0.0))
        confidence = float(trade.get("confidence", 0.0))
        setup_quality = float(trade.get("setup_quality", 0.5))
        vix = float(market.get("vix", 20.0))
        realized_vol = float(market.get("realized_vol", 0.02))
        trade_type = int(trade.get("trade_type", 0))  # 0 = Day, 1 = Swing
        total_signals_today = int(trade.get("total_signals_today", 0))

        # Indicator values with defaults
        def get_safe(val, default=0.0):
            return float(val) if pd.notna(val) else default

        features = np.array([
            pnl,
            confidence,
            setup_quality,
            vix,
            realized_vol,
            trade_type,
            total_signals_today,
            get_safe(row.get("EMA_20")),
            get_safe(row.get("RSI_14"), 50.0),
            get_safe(row.get("MACD")),
            get_safe(row.get("MACD_signal")),
            get_safe(row.get("MACD_hist")),
            get_safe(row.get("BB_upper")),
            get_safe(row.get("BB_middle")),
            get_safe(row.get("BB_lower")),
            get_safe(row.get("VWAP", close)),
            get_safe(row.get("ATR_14")),
            get_safe(row.get("ADX_14"))
        ], dtype=np.float32)

        label = 1 if pnl > 5 else 0  # Good trade threshold

        return features, label, timestamp

    except Exception as e:
        bot_logger.warning(f"[Feature Extract] Skipping entry due to error: {e}")
        return None


def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Log file not found: {META_LOG_PATH}")
        return

    features = []
    labels = []
    timestamps = []

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