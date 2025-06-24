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
    "adx_14",
    "regime_class",
    "classifier_prob",
    "classifier_pred_up",
    "classifier_pred_down",
    "classifier_pred_flat",
    "class_prob_0",
    "class_prob_1",
    "class_prob_2",
    "classifier_entropy",
    "agent_confidence",           # <-- NEW
    "classifier_confidence"       # <-- NEW
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

    open_, high, low, close = bar.get("open"), bar.get("high"), bar.get("low"), bar.get("close")
    volume = bar.get("volume", 1.0)

    if None in (open_, high, low, close):
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

        def safe_val(name: str, fallback: float = 0.0):
            val = row.get(name)
            return fallback if pd.isna(val) else float(val)

        # Label based on PnL
        pct_pnl = float(trade.get("pct_pnl", 0.0))
        if pct_pnl < -2:
            label = 0
        elif pct_pnl > 5:
            label = 1
        else:
            return None  # ambiguous

        confidence_raw = float(trade.get("confidence", 0.0))
        setup_quality_raw = float(trade.get("setup_quality", 0.5))
        vix_raw = float(market.get("vix", 20.0))
        realized_vol_raw = float(market.get("realized_vol", 0.02))

        confidence = normalize(confidence_raw, 0.0, 1.0)
        setup_quality = normalize(setup_quality_raw, 0.0, 1.0)
        vix = normalize(vix_raw, 12.0, 40.0)
        realized_vol = normalize(realized_vol_raw, 0.01, 0.1)

        trade_type = int(trade.get("trade_type", 0))
        total_signals_today = int(trade.get("total_signals_today", 0))
        vwap_value = safe_val("VWAP", fallback=close)

        regime_class = int(classifier.get("regime_class", 1))
        regime_class = max(0, min(regime_class, 3))

        class_probs = classifier.get("class_probabilities", [])
        classifier_prob = float(classifier.get("probability", 0.5))
        classifier_pred = int(classifier.get("predicted_class", 1))
        classifier_entropy = calc_entropy(class_probs) if len(class_probs) == 3 else 1.0

        if len(class_probs) != 3 or classifier_pred not in (0, 1, 2):
            return None

        prob_0, prob_1, prob_2 = map(float, class_probs)
        pred_up = 1.0 if classifier_pred == 0 else 0.0
        pred_down = 1.0 if classifier_pred == 1 else 0.0
        pred_flat = 1.0 if classifier_pred == 2 else 0.0

        # NEW: agent vs classifier confidence
        agent_confidence = normalize(float(trade.get("confidence", 0.5)), 0.0, 1.0)
        classifier_confidence = normalize(float(classifier.get("prob", 0.5)), 0.0, 1.0)

        features = np.array([
            confidence,
            setup_quality,
            vix,
            realized_vol,
            trade_type,
            total_signals_today,
            safe_val("EMA_20"),
            safe_val("RSI_14", 50.0),
            safe_val("MACD"),
            safe_val("MACD_signal"),
            safe_val("MACD_hist"),
            safe_val("BB_upper"),
            safe_val("BB_middle"),
            safe_val("BB_lower"),
            vwap_value,
            safe_val("ATR_14"),
            safe_val("ADX_14"),
            regime_class,
            classifier_prob,
            pred_up,
            pred_down,
            pred_flat,
            prob_0,
            prob_1,
            prob_2,
            classifier_entropy,
            agent_confidence,
            classifier_confidence
        ], dtype=np.float32)

        return features, label, timestamp

    except Exception as e:
        bot_logger.warning(f"[Feature Extract] Skipping entry: {e}")
        return None

def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Missing: {META_LOG_PATH}")
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
                bot_logger.warning(f"[Build Dataset] Malformed JSON at line {line_num}")

    if not features:
        bot_logger.error("[Build Dataset] No valid features extracted.")
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

    # Backup old CSV
    if OUTPUT_CSV.exists():
        backup_path = OUTPUT_CSV.with_name(f"spy_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(OUTPUT_CSV, backup_path)
        bot_logger.info(f"[Build Dataset] Backup saved: {backup_path}")

    # Save final dataset
    df_combined.to_csv(OUTPUT_CSV, index=False)
    np.savez_compressed(OUTPUT_NPZ,
        X=df_combined[FEATURE_NAMES].values.astype(np.float32),
        y=df_combined["label"].values.astype(np.int32),
        timestamps=df_combined["timestamp"].values
    )

    bot_logger.info(f"[Build Dataset] ✅ {len(df_new)} new, {len(df_combined)} total | pos={pos}, neg={neg}, skip={skip}")