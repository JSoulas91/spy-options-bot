import os
import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils.telegram import send_telegram_message, send_telegram_file

# Paths
DATA_PATH = "ml/spy_data.csv"
RAW_MODEL_PATH = "models/xgb_raw.json"
CAL_MODEL_PATH = "models/xgb_calibrated.pkl"
PLOT_PATH = "ml/calibration_plot.png"
ACCURACY_LOG_PATH = "ml/accuracy_log.txt"
MAX_ROWS = 10000

LOG = logging.getLogger("retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s")

def backup_model(src_path, prefix):
    if os.path.exists(src_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = f"models/{prefix}_{ts}" + (".pkl" if src_path.endswith(".pkl") else ".json")
        os.rename(src_path, dst)
        LOG.info(f"[Backup] Backed up model to {dst}")

def load_data(path):
    df = pd.read_csv(path)
    LOG.info(f"[Load Data] Loaded {len(df)} rows")
    return df

def prune_training_data(df):
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)
        LOG.info(f"[Data Prune] Limited to last {MAX_ROWS} rows")
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
        LOG.info("[ML Retraining] Started")
        df = load_data(DATA_PATH)
        df = df.dropna(subset=["label"])
        df = prune_training_data(df)

        X = df.drop(columns=["timestamp", "label", "pnl"], errors="ignore")
        y = df["label"]

        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        LOG.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        with open(ACCURACY_LOG_PATH, "a") as f:
            f.write(f"{acc:.4f}\n")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv=skf)
        calibrator.fit(X, y)
        probs = calibrator.predict_proba(X)[:, 1]
        save_calibration_plot(probs, y, PLOT_PATH)

        os.makedirs("models", exist_ok=True)
        backup_model(RAW_MODEL_PATH, "xgb_raw")
        backup_model(CAL_MODEL_PATH, "xgb_calibrated")

        if hasattr(calibrator, "calibrated_classifiers_"):
            raw_model = calibrator.calibrated_classifiers_[0].estimator
            raw_model.save_model(RAW_MODEL_PATH)
            LOG.info(f"[Model Saved] Raw XGBoost model saved to {RAW_MODEL_PATH}")

        joblib.dump(calibrator, CAL_MODEL_PATH)
        LOG.info(f"[Model Saved] Calibrated model saved to {CAL_MODEL_PATH}")

        send_telegram_message(f"✅ ML retraining complete.\nAccuracy: {acc:.4f}")
        # send_telegram_file(PLOT_PATH)

    except Exception as e:
        LOG.critical(f"Fatal error: {e}", exc_info=True)
        send_telegram_message(f"❌ ML retraining failed:\n{e}")

if __name__ == "__main__":
    retrain_model()