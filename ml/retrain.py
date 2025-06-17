import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from xgboost import Booster
from build_spy_data_from_meta_log import build_dataset
from utils.logger import bot_logger
from utils.telegram_utils import send_telegram_message
from utils.health_check import update_status
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

MODEL_PATH = Path("models/xgb_raw.json")
CALIBRATED_MODEL_PATH = Path("models/xgb_calibrated.json")
DATA_PATH = Path("ml/spy_data.csv")
ACCURACY_LOG = Path("ml/accuracy_log.txt")

logger = bot_logger

def safe_telegram_message(text: str):
    """Escape markdown special chars or remove problematic chars for Telegram."""
    replacements = {
        '_': '\\_',
        '*': '\\*',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '~': '\\~',
        '`': '\\`',
        '>': '\\>',
        '#': '\\#',
        '+': '\\+',
        '-': '\\-',
        '=': '\\=',
        '|': '\\|',
        '{': '\\{',
        '}': '\\}',
        '.': '\\.',
        '!': '\\!'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def load_data():
    if not DATA_PATH.exists():
        logger.error(f"[Load Data] Data file not found: {DATA_PATH}")
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Filter or clean if necessary here
    # Drop columns not used as features or label
    features = df.drop(columns=["timestamp", "label"], errors='ignore')
    labels = df["label"]
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(features.columns)}")
    return features, labels

def retrain_model():
    logger.info("[ML Retraining] Started")

    # Build latest dataset (append new data and prune)
    logger.info("[Preprocess] Running feature builder...")
    build_dataset()
    logger.info("[Preprocess] Feature builder completed successfully.")

    # Load dataset
    X, y = load_data()

    try:
        # Train XGBoost classifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X, y)

        # In-sample accuracy
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        # Save raw model booster
        model.get_booster().save_model(str(MODEL_PATH))
        logger.info(f"[Save Model] Raw XGBoost booster saved to {MODEL_PATH}")

        # Calibration with isotonic regression using cross-validation
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv=5)
        calibrator.fit(X, y)
        # Save the underlying base estimator's booster after calibration
        base_estimator = calibrator.base_estimator_
        base_estimator.get_booster().save_model(str(MODEL_PATH))  # overwrite raw model file with calibrated base

        # Save calibrated model separately using joblib or pickle
        import joblib
        joblib.dump(calibrator, CALIBRATED_MODEL_PATH.with_suffix(".joblib"))
        logger.info(f"[Save Model] Calibrated model saved to {CALIBRATED_MODEL_PATH.with_suffix('.joblib')}")

        # Log accuracy to file with timestamp
        from datetime import datetime
        with open(ACCURACY_LOG, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} - accuracy: {acc:.4f}\n")

        # Send Telegram notification
        message = f"✅ ML Retraining completed.\nIn-sample accuracy: {acc:.4f}"
        send_telegram_message(safe_telegram_message(message))

        update_status("ml_retrain")  # success, no extra args

    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(f"Fatal error during retraining: {e}\n{tb}")
        try:
            send_telegram_message(f"❌ ML Retraining failed:\n{safe_telegram_message(str(e))}")
        except Exception:
            logger.error("Failed to send Telegram failure message.")
        update_status("ml_retrain", status="fail")  # pass one arg keyword-style if your update_status supports it

if __name__ == "__main__":
    retrain_model()