import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import bot_logger
from technical_analysis.indicators import calculate_indicators

META_LOG_PATH = Path("meta/meta_log.jsonl")
OUTPUT_NPZ = Path("ml/dataset.npz")
OUTPUT_CSV = Path("ml/spy_data.csv")
MAX_ROWS = 30000

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
    "adx_14",
    "trade_success_prob",
    "predicted_direction",
    "classifier_entropy"
]

def extract_features(entry: dict) -> tuple[np.ndarray, int, str] | None:
    trade = entry.get("trade", {})
    bar = entry.get("bar", {})
    market = entry.get("market", {})
    classifier = trade.get("classifier", {})
    timestamp = entry.get("timestamp")

    open_, high, low, close = bar.get("open"), bar.get("high"), bar.get("low"), bar.get("close")
    volume = bar.get("volume", 1.0)

    if None in (open_, high, low, close):
        bot_logger.warning("[Feature Extract] Skipping entry: Missing OHLC data")
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
            if pd.isna(val):
                bot_logger.debug(f"[Indicator] Missing {name}, using fallback={fallback}")
                return fallback
            return float(val)

        pnl = float(trade.get("pnl", 0.0))
        confidence = float(trade.get("confidence", 0.0))
        setup_quality = float(trade.get("setup_quality", 0.5))
        vix = float(market.get("vix", 20.0))
        realized_vol = float(market.get("realized_vol", 0.02))
        trade_type = int(trade.get("trade_type", 0))
        total_signals_today = int(trade.get("total_signals_today", 0))
        vwap_value = safe_val("VWAP", fallback=close)

        trade_success_prob = float(classifier.get("trade_success_prob", 0.0))
        predicted_direction = int(classifier.get("predicted_direction", -1))
        classifier_entropy = float(classifier.get("entropy", 0.0))

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
            safe_val("ADX_14"),
            trade_success_prob,
            predicted_direction,
            classifier_entropy
        ], dtype=np.float32)

        label = 1 if pnl > 5 else 0
        return features, label, timestamp

    except Exception as e:
        bot_logger.warning(f"[Feature Extract] Skipping entry due to exception: {e}")
        return None

def build_dataset():
    if not META_LOG_PATH.exists():
        bot_logger.error(f"[Build Dataset] Log file not found: {META_LOG_PATH}")
        return

    features, labels, timestamps = [], [], []
    skipped = 0

    with META_LOG_PATH.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                result = extract_features(entry)
                if result:
                    feat, label, ts = result
                    features.append(feat)
                    labels.append(label)
                    timestamps.append(ts)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
                bot_logger.warning(f"[Build Dataset] Skipping malformed JSON at line {line_num}")

    if len(features) == 0:
        bot_logger.error("[Build Dataset] No valid entries found.")
        return

    df_new = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_new.insert(0, "timestamp", timestamps)
    df_new["label"] = labels

    if OUTPUT_CSV.exists():
        df_existing = pd.read_csv(OUTPUT_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.sort_values("timestamp")
    else:
        df_combined = df_new

    if len(df_combined) > MAX_ROWS:
        df_combined = df_combined.tail(MAX_ROWS)

    df_combined.to_csv(OUTPUT_CSV, index=False)
    np.savez_compressed(OUTPUT_NPZ,
        X=df_combined[FEATURE_NAMES].values.astype(np.float32),
        y=df_combined["label"].values.astype(np.int32),
        timestamps=df_combined["timestamp"].values
    )

    bot_logger.info(f"[Build Dataset] ✅ Appended {len(df_new)} new entries. Total rows: {len(df_combined)}")
    bot_logger.info(f"[Build Dataset] Skipped {skipped} entries due to missing data or errors.")

    # 🔍 Debug print summary
    print(f"\n[Debug] Final dataset shape: {df_combined.shape}")
    print(f"[Debug] Feature columns: {list(df_combined.columns)}")
    print("[Debug] Sample rows:\n", df_combined.head(5))

if __name__ == "__main__":
    build_dataset()