import json
import os
import numpy as np
from pathlib import Path
from utils.logger import bot_logger

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_DATASET = Path("ml/dataset.npz")

def extract_features(entry: dict) -> tuple[np.ndarray, int] | None:
    """
    Transform a meta log entry into ML features and binary label (1 = good trade).
    Returns None if trade is invalid or unusable.
    """
    trade = entry.get("trade", {})
    market = entry.get("market", {})

    try:
        pnl = float(trade.get("pnl", 0))
        confidence = float(trade.get("confidence", 0))
        setup_quality = float(trade.get("setup_quality", 0))
        vix = float(market.get("vix", 15))
        realized_vol = float(market.get("realized_vol", 1.0))
        trade_type = int(trade.get("trade_type", 0))  # 0=Day, 1=Swing
        total_signals_today = int(trade.get("total_signals_today", 10))

        features = np.array([
            pnl,
            confidence,
            setup_quality,
            vix,
            realized_vol,
            trade_type,
            total_signals_today
        ], dtype=np.float32)

        label = 1 if pnl > 5 else 0  # Label good trades as 1

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

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if len(features) == 0:
        bot_logger.error("[Build Dataset] No valid entries found.")
        return

    np.savez_compressed(OUTPUT_DATASET, X=features, y=labels)
    bot_logger.info(f"[Build Dataset] ✅ Saved {len(features)} entries to {OUTPUT_DATASET}")

if __name__ == "__main__":
    build_dataset()