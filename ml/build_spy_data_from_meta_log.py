import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import apply_indicators

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_DATASET = Path("ml/dataset.npz")
DEBUG_DF_OUTPUT = Path("ml/spy_data.csv")  # Optional for inspection

def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Log file not found: {META_LOG_PATH}")
        return

    rows = []
    with META_LOG_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                trade = entry.get("trade", {})
                market = entry.get("market", {})
                timestamp = entry.get("timestamp")

                pnl = float(trade.get("pnl", 0))
                confidence = float(trade.get("confidence", 0))
                setup_quality = float(trade.get("setup_quality", 0))
                vix = float(market.get("vix", 15))
                realized_vol = float(market.get("realized_vol", 1.0))
                trade_type = int(trade.get("trade_type", 0))  # 0=Day, 1=Swing
                total_signals_today = int(trade.get("total_signals_today", 10))
                close = float(market.get("close", np.nan))
                high = float(market.get("high", np.nan))
                low = float(market.get("low", np.nan))
                open_ = float(market.get("open", np.nan))
                volume = float(market.get("volume", 1.0))

                row = {
                    "timestamp": timestamp,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vix": vix,
                    "confidence": confidence,
                    "setup_quality": setup_quality,
                    "realized_vol": realized_vol,
                    "trade_type": trade_type,
                    "total_signals_today": total_signals_today,
                    "pnl": pnl,
                    "label": 1 if pnl > 5 else 0,
                }

                rows.append(row)

            except Exception as e:
                bot_logger.warning(f"[Feature Extract] Skipping entry due to error: {e}")
                continue

    if not rows:
        bot_logger.error("[Build Dataset] No valid entries parsed.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = apply_indicators(df)

    # Save raw feature matrix + labels
    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X = df[feature_cols].astype(np.float32).values
    y = df["label"].astype(np.int32).values

    np.savez_compressed(OUTPUT_DATASET, X=X, y=y)
    bot_logger.info(f"[Build Dataset] ✅ Saved {len(X)} entries to {OUTPUT_DATASET}")

    # Optional CSV output for debugging
    df.to_csv(DEBUG_DF_OUTPUT, index=False)
    bot_logger.info(f"[Build Dataset] 🔍 Saved preview to {DEBUG_DF_OUTPUT}")

if __name__ == "__main__":
    build_dataset()