import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
import logging

# === Constants ===
META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_CSV = Path("ml/spy_data.csv")
OUTPUT_NPZ = Path("ml/spy_data.npz")
MAX_ROWS = 50000

# === 17 Expected Feature Keys ===
FEATURE_NAMES = [
    "confidence", "setup_quality", "vix", "realized_vol", "trade_type",
    "total_signals_today", "ema_20", "rsi_14", "macd", "macd_signal",
    "macd_hist", "bb_upper", "bb_middle", "bb_lower", "vwap", "atr_14", "adx_14"
]

# === Logger Setup ===
logger = logging.getLogger("build_spy_data_from_meta_log")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
bot_logger = logger


# === Feature Extractor ===
def extract_features(entry: dict):
    try:
        if "features" not in entry or not isinstance(entry["features"], dict):
            bot_logger.warning(f"[Skip] Missing or malformed 'features' field: {entry.get('features')}")
            return None

        features = entry["features"]
        timestamp = entry.get("timestamp")
        pct_pnl = entry.get("pct_pnl")

        if timestamp is None or pct_pnl is None:
            bot_logger.warning(f"[Skip] Missing timestamp or pct_pnl: keys={entry.keys()}")
            return None

        # Ensure all required feature keys are present
        if not all(k in features for k in FEATURE_NAMES):
            bot_logger.warning(f"[Skip] Missing keys in features: {features}")
            return None

        feature_vector = [features[k] for k in FEATURE_NAMES]
        label = 1 if pct_pnl >= 0 else 0

        return feature_vector, label, timestamp
    except Exception as e:
        bot_logger.exception(f"[extract_features] Exception: {e}")
        return None


# === Main Dataset Builder ===
def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Missing meta log: {META_LOG_PATH}")
        return

    features, labels, timestamps = [], [], []
    pos, neg, skip = 0, 0, 0

    with META_LOG_PATH.open("r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                result = extract_features(entry)
                if result:
                    feat, label, ts = result
                    features.append(feat)
                    labels.append(label)
                    timestamps.append(ts)
                    if label == 1:
                        pos += 1
                    else:
                        neg += 1
                else:
                    skip += 1
            except json.JSONDecodeError:
                bot_logger.warning(f"[Build Dataset] JSON parse error at line {line_num}")
                skip += 1

    if not features:
        bot_logger.error("[Build Dataset] No features extracted.")
        return

    df_new = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_new.insert(0, "timestamp", timestamps)
    df_new["label"] = labels

    if OUTPUT_CSV.exists():
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.sort_values("timestamp").drop_duplicates(subset="timestamp")
    else:
        df_combined = df_new

    if len(df_combined) > MAX_ROWS:
        df_combined = df_combined.tail(MAX_ROWS)

    if OUTPUT_CSV.exists():
        backup_path = OUTPUT_CSV.with_name(f"spy_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(OUTPUT_CSV, backup_path)
        bot_logger.info(f"[Backup] Existing CSV backed up: {backup_path}")

    df_combined.to_csv(OUTPUT_CSV, index=False)
    np.savez_compressed(OUTPUT_NPZ,
        X=df_combined[FEATURE_NAMES].values.astype(np.float32),
        y=df_combined["label"].values.astype(np.int32),
        timestamps=df_combined["timestamp"].values
    )

    bot_logger.info(f"[✅ Build Complete] +{len(df_new)} new rows, {len(df_combined)} total | pos={pos}, neg={neg}, skip={skip}")


# === Entry Point ===
if __name__ == "__main__":
    build_dataset()