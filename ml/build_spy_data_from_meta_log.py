import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import compute_trade_indicators

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_NPZ = Path("ml/dataset.npz")
OUTPUT_CSV = Path("ml/spy_data.csv")

def extract_features(entry: dict) -> tuple[np.ndarray, int] | None:
    trade = entry.get("trade", {})
    market = entry.get("market", {})
    prices = entry.get("prices", {})

    if not prices or not all(k in prices for k in ("open", "high", "low", "close", "volume")):
        bot_logger.warning("[Feature Extract] Skipping entry due to missing price data")
        return None

    try:
        df = pd.DataFrame(prices)
        indicators = compute_trade_indicators(df)

        pnl = float(trade.get("pnl", 0))
        confidence = float(trade.get("confidence", 0))
        setup_quality = float(trade.get("setup_quality", 0))
        vix = float(market.get("vix", 15))
        realized_vol = float(market.get("realized_vol", 1.0))
        trade_type = int(trade.get("trade_type", 0))  # 0=Day, 1=Swing
        total_signals_today = int(trade.get("total_signals_today", 10))

        # Select most recent row of indicators
        latest = indicators.iloc[-1]

        features = np.array([
            pnl,
            confidence,
            setup_quality,
            vix,
            realized_vol,
            trade_type,
            total_signals_today,
            latest["EMA_20"],
            latest["RSI_14"],
            latest["MACD"],
            latest["MACD_signal"],
            latest["MACD_hist"],
            latest["BB_upper"],
            latest["BB_middle"],
            latest["BB_lower"],
            latest["VWAP"],
            latest["ATR_14"],
            latest["ADX_14"]
        ], dtype=np.float32)

        label = 1 if pnl > 5 else 0  # Good trades only

        return features, label

    except Exception as e:
        bot_logger.warning(f"[Feature Extract] Skipping entry due to error: {e}")
        return None

def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Log file not found: {META_LOG_PATH}")
        return

    features = []
    labels = []

    with META_LOG_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                result = extract_features(entry)
                if result:
                    feat, label = result
                    features.append(feat)
                    labels.append(label)
            except json.JSONDecodeError:
                continue

    if len(features) == 0:
        bot_logger.error("[Build Dataset] No valid entries found.")
        return

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    np.savez_compressed(OUTPUT_NPZ, X=features, y=labels)
    df_out = pd.DataFrame(features)
    df_out["label"] = labels
    df_out.to_csv(OUTPUT_CSV, index=False)

    bot_logger.info(f"[Build Dataset] ✅ Saved {len(features)} entries to {OUTPUT_NPZ} and {OUTPUT_CSV}")

if __name__ == "__main__":
    build_dataset()