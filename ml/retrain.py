import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from ml.build_spy_data_from_meta_log import build_dataset
from utils.logger import bot_logger
from utils.telegram_utils import send_telegram_message
from monitor.health_check import update_status
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

MODEL_PATH = Path("models/xgb_raw.json")
CALIBRATED_MODEL_PATH = Path("models/xgb_calibrated.pkl")
CSV_PATH = Path("ml/spy_data.csv")
ACCURACY_LOG_PATH = Path("ml/accuracy_log.txt")

FEATURE_COLS = [
    "pnl", "confidence", "setup_quality", "vix", "realized_vol", "trade_type", "total_signals_today",
    "ema_20", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "vwap", "atr_14", "adx_14"
]

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y, df

def retrain_model():
    try:
        bot_logger.info("[ML Retraining] Started")

        # Step 1: Build or update dataset
        bot_logger.info("[Preprocess] Running feature builder...")
        build_dataset()
        bot_logger.info("[Preprocess] Feature builder completed successfully.")

        # Step 2: Load dataset
        X, y, df = load_data()
        bot_logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")

        # Step 3: Train XGBoost model
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        xgb.fit(X, y)

        # Save raw model
        xgb.save_model(str(MODEL_PATH))
        bot_logger.info(f"[Save Model] Raw XGBoost booster saved to {MODEL_PATH}")

        # Step 4: Calibrate model with 5-fold isotonic calibration
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(base_estimator=xgb, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        # Save calibrated model
        joblib.dump(calibrator, CALIBRATED_MODEL_PATH)
        bot_logger.info(f"[Save Model] Calibrated model saved to {CALIBRATED_MODEL_PATH}")

        # Step 5: Evaluate accuracy
        y_pred = cross_val_predict(calibrator, X, y, cv=skf, method='predict')
        acc = accuracy_score(y, y_pred)
        bot_logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        # Step 6: Log accuracy
        with open(ACCURACY_LOG_PATH, "a") as f:
            f.write(f"{pd.Timestamp.now()},rows={len(df)},accuracy={acc:.4f}\n")

        # Step 7: Send Telegram alert
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(
                f"✅ Model retrained ({len(df)} rows)\n"
                f"Accuracy: {acc:.4f}\n"
                f"Saved to: `{MODEL_PATH.name}`, `{CALIBRATED_MODEL_PATH.name}`",
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID
            )

        update_status("ml_retrain", "ok")

    except Exception as e:
        bot_logger.critical(f"Fatal error during retraining: {e}")

        # Notify via Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(f"❌ ML retrain failed: {e}", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

        update_status("ml_retrain", "fail")

if __name__ == "__main__":
    retrain_model()