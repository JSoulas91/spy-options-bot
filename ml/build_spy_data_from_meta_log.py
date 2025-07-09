import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import calculate_indicators
from datetime import datetime
import shutil
import math

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_CSV = Path("ml/spy_data.csv")
OUTPUT_NPZ = Path("ml/dataset.npz")
MAX_ROWS = 30000

FEATURE_NAMES = [
    "confidence", "setup_quality", "vix", "realized_vol", "trade_type", "total_signals_today",
    "ema_20", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "vwap", "atr_14", "adx_14", "regime_class",
    "classifier_prob", "classifier_pred_up", "classifier_pred_down", "classifier_pred_flat",
    "class_prob_0", "class_prob_1", "class_prob_2",
    "classifier_entropy", "agent_confidence", "classifier_confidence"
]

def normalize(val, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))

def calc_entropy(probs: list[float]) -> float:
    eps = 1e-8
    return -sum(p * math.log(p + eps) for p in probs)

def extract_features(entry: dict) -> tuple[np.ndarray, int, str] | None:
    trade = entry.get("trade", {})
    bar = entry.get("bar", {})
    market = entry.get("market", {})
    classifier = trade.get("classifier", {})
    timestamp = entry.get("timestamp")

    if not all(k in bar for k in ("open", "high", "low", "close", "volume")):
        bot_logger.warning(f"[Skip] Missing OHLCV data @ {timestamp}")
        return None

    try:
        df_bar = pd.DataFrame([{
            'open': float(bar['open']),
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
            'volume': float(bar['volume']),
        }])
        df_ind = calculate_indicators(df_bar)
        row = df_ind.iloc[0]

        for ind in ["EMA_20", "RSI_14", "MACD", "MACD_signal", "MACD_hist",
                    "BB_upper", "BB_middle", "BB_lower", "VWAP", "ATR_14", "ADX_14"]:
            if pd.isna(row.get(ind)):
                bot_logger.warning(f"[Skip] Missing indicator '{ind}' @ {timestamp}")
                return None

        pct_pnl = trade.get("pct_pnl")
        if pct_pnl is None:
            bot_logger.warning(f"[Skip] Missing pct_pnl @ {timestamp}")
            return None
        pct_pnl = float(pct_pnl)
        if pct_pnl < -1:
            label = 0
        elif pct_pnl > 2:
            label = 1
        else:
            bot_logger.debug(f"[Skip] Ambiguous PnL @ {timestamp}: {pct_pnl}")
            return None

        for key in ["confidence", "setup_quality", "trade_type", "total_signals_today"]:
            if key not in trade:
                bot_logger.warning(f"[Skip] Missing trade.{key} @ {timestamp}")
                return None

        for key in ["vix", "realized_vol"]:
            if key not in market:
                bot_logger.warning(f"[Skip] Missing market.{key} @ {timestamp}")
                return None

        class_probs = classifier.get("class_probabilities")
        if not isinstance(class_probs, list) or len(class_probs) != 3:
            bot_logger.warning(f"[Skip] Invalid class_probabilities @ {timestamp}: {class_probs}")
            return None

        classifier_pred = classifier.get("predicted_class")
        if classifier_pred not in (0, 1, 2):
            bot_logger.warning(f"[Skip] Invalid predicted_class @ {timestamp}: {classifier_pred}")
            return None

        features = np.array([
            normalize(float(trade["confidence"]), 0.0, 1.0),
            normalize(float(trade["setup_quality"]), 0.0, 1.0),
            normalize(float(market["vix"]), 12.0, 40.0),
            normalize(float(market["realized_vol"]), 0.01, 0.1),
            int(trade["trade_type"]),
            int(trade["total_signals_today"]),
            float(row["EMA_20"]),
            float(row["RSI_14"]),
            float(row["MACD"]),
            float(row["MACD_signal"]),
            float(row["MACD_hist"]),
            float(row["BB_upper"]),
            float(row["BB_middle"]),
            float(row["BB_lower"]),
            float(row["VWAP"]),
            float(row["ATR_14"]),
            float(row["ADX_14"]),
            int(classifier.get("regime_class", 1)),
            float(classifier.get("probability", 0.5)),
            1.0 if classifier_pred == 0 else 0.0,
            1.0 if classifier_pred == 1 else 0.0,
            1.0 if classifier_pred == 2 else 0.0,
            float(class_probs[0]),
            float(class_probs[1]),
            float(class_probs[2]),
            calc_entropy(class_probs),
            normalize(float(trade["confidence"]), 0.0, 1.0),
            normalize(float(classifier.get("prob", 0.5)), 0.0, 1.0),
        ], dtype=np.float32)

        return features, label, timestamp

    except Exception as e:
        bot_logger.error(f"[Extract Error @ {timestamp}] {e}", exc_info=True)
        return None

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
    
    if __name__ == "__main__":
    build_dataset()