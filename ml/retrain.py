import os
import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import subprocess
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.telegram_utils import send_telegram_message, send_plot as send_telegram_file
from utils.logger import bot_logger as logger
from monitor.health_check import update_status

# Constants
DATA_PATH = "ml/spy_data.csv"
RAW_MODEL_PATH = "models/xgb_raw.json"
CAL_MODEL_PATH = "models/xgb_calibrated.pkl"
PLOT_PATH = "ml/calibration_plot.png"
ACCURACY_LOG_PATH = "ml/accuracy_log.txt"
MAX_ROWS = 10000


def run_build_dataset():
    try:
        logger.info("[Preprocess] Running feature builder...")
        subprocess.run(["python3", "ml/build_spy_data_from_meta_log.py"], check=True)
        logger.info("[Preprocess] Feature builder completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"[Preprocess] Feature builder failed: {e}")
        send_telegram_message("❌ Feature extraction failed before ML retraining.")
        raise


def load_data(path):
    df = pd.read_csv(path)
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def prune_training_data(df):
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)
        logger.info(f"[Data Prune] Limited to last {MAX_ROWS} rows")
    return df


def save_calibration_plot(probs, y_true, path):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy='uniform')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("True Proportion of Positives")
    plt.title("Calibration Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def retrain_model():
    try:
        logger.info("[ML Retraining] Started")

        run_build_dataset()

        df = load_data(DATA_PATH)
        df = df.dropna(subset=["label"])
        df = prune_training_data(df)

        label = df["label"]
        X = df.drop(columns=["timestamp", "label", "pnl"], errors="ignore")

        # Raw model
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0)
        model.fit(X, label)
        y_pred = model.predict(X)
        acc = accuracy_score(label, y_pred)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        with open(ACCURACY_LOG_PATH, "a") as f:
            f.write(f"{acc:.4f}\n")

        # Calibrate
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv=skf)
        calibrator.fit(X, label)

        probs = calibrator.predict_proba(X)[:, 1]
        save_calibration_plot(probs, label, PLOT_PATH)

        os.makedirs(os.path.dirname(CAL_MODEL_PATH), exist_ok=True)

        # Save raw model
        try:
            if hasattr(calibrator, "calibrated_classifiers_"):
                raw_model = calibrator.calibrated_classifiers_[0].estimator
                raw_model.save_model(RAW_MODEL_PATH)
                logger.info(f"[Model Saved] Raw XGBoost model saved to {RAW_MODEL_PATH}")
            else:
                logger.warning("[Model] Could not extract raw model from calibrator.")
        except Exception as e:
            logger.warning(f"[Model] Failed to save raw model: {e}")

        # Save calibrated model
        joblib.dump(calibrator, CAL_MODEL_PATH)
        logger.info(f"[Model Saved] Calibrated model saved to {CAL_MODEL_PATH}")

        # Backup copy with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = f"models/xgb_calibrated_{ts}.pkl"
        joblib.dump(calibrator, backup_path)
        logger.info(f"[Backup] Timestamped model backup saved to {backup_path}")

        # Telegram
        send_telegram_message(f"✅ ML retraining complete.\nAccuracy: {acc:.4f}")
        # send_telegram_file(PLOT_PATH)

        # Health check
        update_status("ml_retrain:ok")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}", exc_info=True)
        send_telegram_message(f"❌ ML retraining failed:\n{e}")
        update_status("ml_retrain:fail")


if __name__ == "__main__":
    retrain_model()