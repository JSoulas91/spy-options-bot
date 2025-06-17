import os
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ml.build_spy_data_from_meta_log import build_dataset
from monitor.health_check import update_status
from utils.telegram_utils import send_telegram_message
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s"
)
logger = logging.getLogger("retrain")

DATA_CSV = Path("ml/spy_data.csv")
RAW_MODEL_PATH = Path("models/xgb_raw.json")
CAL_MODEL_PATH = Path("models/xgb_calibrated.pkl")
ACCURACY_LOG = Path("ml/accuracy_log.txt")


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna()
    y = df["label"].astype(int).values
    X = df.drop(columns=["timestamp", "label", "pnl"]).values.astype(np.float32)
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return X, y


def retrain_model():
    logger.info("[ML Retraining] Started")

    try:
        logger.info("[Preprocess] Running feature builder...")
        build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")
    except Exception as e:
        logger.exception(f"Feature builder failed: {e}")
        send_telegram_message(f"❌ Feature builder failed: {e}")
        update_status("ml_retrain", "fail")
        return

    try:
        X, y = load_data()

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            verbosity=0
        )
        xgb.fit(X, y)

        # Save raw model
        xgb.save_model(RAW_MODEL_PATH)
        logger.info(f"[Save Model] Raw XGBoost booster saved to {RAW_MODEL_PATH}")

        # Calibration
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv=skf)
        calibrator.fit(X, y)
        joblib.dump(calibrator, CAL_MODEL_PATH)
        logger.info(f"[Save Model] Calibrated model saved to {CAL_MODEL_PATH}")

        accuracy = calibrator.score(X, y)
        logger.info(f"[Accuracy] In-sample accuracy: {accuracy:.4f}")
        with ACCURACY_LOG.open("a") as f:
            f.write(f"{datetime.utcnow().isoformat()} — Accuracy: {accuracy:.4f}\n")

        send_telegram_message(f"✅ ML retrained. Accuracy: {accuracy:.4f}")
        update_status("ml_retrain", "ok")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}")
        send_telegram_message(f"❌ ML retrain failed: {e}")
        update_status("ml_retrain", "fail")


if __name__ == "__main__":
    retrain_model()