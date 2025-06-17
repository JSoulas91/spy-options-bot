import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

from utils.telegram_utils import send_telegram_message
from monitor.health_check import update_status
from build_spy_data_from_meta_log import run_build_dataset

# Constants
DATA_PATH = "ml/spy_data.csv"
MODEL_PATH = "models/xgb_raw.json"
BACKUP_DIR = "models/backups"
ACCURACY_LOG = "ml/accuracy_log.txt"

# Setup logger
logger = logging.getLogger("retrain")
logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s",
    level=logging.INFO,
)


class XGBWrapper:
    """Wrapper for XGBClassifier with raw model save/load support."""

    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)


def load_data():
    logger.info(f"[Load Data] Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Drop any rows with NaNs if necessary
    df = df.dropna()
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Define features and label
    features = [
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
    ]

    X = df[features].values
    y = df["label"].values
    return X, y, df


def retrain_model():
    logger.info("[ML Retraining] Started")
    try:
        # Step 1: Build dataset (append new data)
        logger.info("[Preprocess] Running feature builder...")
        run_build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")

        # Step 2: Load data for training
        X, y, df = load_data()

        # Step 3: Setup model and stratified folds for calibration
        model = XGBWrapper(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.1,
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Step 4: Calibrated classifier with isotonic calibration
        calibrator = CalibratedClassifierCV(estimator=model.model, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        preds = calibrator.predict(X)
        acc = accuracy_score(y, preds)

        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        # Step 5: Save the underlying fitted XGB model from calibrated classifier
        trained_model = calibrator.calibrated_classifiers_[0].base_estimator
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        trained_model.save_model(MODEL_PATH)

        # Backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_path = os.path.join(BACKUP_DIR, f"xgb_model_{timestamp}.json")
        trained_model.save_model(backup_path)

        # Log accuracy to file
        with open(ACCURACY_LOG, "a") as f:
            f.write(f"{timestamp},{acc:.4f}\n")

        # Telegram alert on success
        send_telegram_message(
            f"✅ <b>Classifier Retrained</b>\n"
            f"<b>Samples:</b> {len(df)}\n"
            f"<b>Accuracy:</b> {acc:.4f}"
        )

        update_status("ml_retrain:success")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}", exc_info=True)

        # Sanitize error message for Telegram
        err_msg = str(e).replace("<", "&lt;").replace(">", "&gt;")
        send_telegram_message(
            f"❌ <b>Retraining failed</b>\n<pre>{err_msg[:1000]}</pre>"
        )

        update_status("ml_retrain:fail")


if __name__ == "__main__":
    retrain_model()