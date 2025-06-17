import os
import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils.telegram import send_telegram_message, send_telegram_file

# Constants
DATA_PATH = "ml/spy_data.csv"
RAW_MODEL_PATH = "models/xgb_raw.json"
CAL_MODEL_PATH = "models/xgb_calibrated.pkl"
PLOT_PATH = "ml/calibration_plot.png"
ACCURACY_LOG_PATH = "ml/accuracy_log.txt"
MAX_ROWS = 10000
LOG = logging.getLogger("retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s")

def load_data(path):
    df = pd.read_csv(path)
    LOG.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
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

        # Drop rows with NaNs
        df = df.dropna(subset=["label"])

        # Use semantic feature columns
        label = df["label"]
        X = df.drop(columns=["timestamp", "label", "pnl"], errors="ignore")

        if "open" not in df.columns:
            LOG.info("[Indicators] Skipped — no OHLCV")

        df = prune_training_data(df)

        # Raw XGBoost model
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0)
        model.fit(X, label)
        y_pred = model.predict(X)
        acc = accuracy_score(label, y_pred)
        LOG.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        with open(ACCURACY_LOG_PATH, "a") as f:
            f.write(f"{acc:.4f}\n")

        # Calibrate model
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv=skf)
        calibrator.fit(X, label)

        # Save calibration plot
        probs = calibrator.predict_proba(X)[:, 1]
        save_calibration_plot(probs, label, PLOT_PATH)

        # Ensure models dir exists
        os.makedirs(os.path.dirname(CAL_MODEL_PATH), exist_ok=True)

        # Save raw internal XGB model from calibrator
        try:
            if hasattr(calibrator, "calibrated_classifiers_"):
                raw_model = calibrator.calibrated_classifiers_[0].estimator
                raw_model.save_model(RAW_MODEL_PATH)
                LOG.info(f"[Model Saved] Raw XGBoost model saved to {RAW_MODEL_PATH}")
            else:
                LOG.warning("[Model] Could not extract raw model — structure unknown.")
        except Exception as e:
            LOG.warning(f"[Model] Could not save raw model: {e}")

        # Save calibrated model
        joblib.dump(calibrator, CAL_MODEL_PATH)
        LOG.info(f"[Model Saved] Calibrated model saved to {CAL_MODEL_PATH}")

        # Telegram success message
        send_telegram_message(f"✅ ML retraining complete.\nAccuracy: {acc:.4f}")
        # send_telegram_file(PLOT_PATH)  # Uncomment if you want to send calibration curve image

    except Exception as e:
        LOG.critical(f"Fatal error: {e}", exc_info=True)
        send_telegram_message(f"❌ ML retraining failed:\n{e}")

if __name__ == "__main__":
    retrain_model()