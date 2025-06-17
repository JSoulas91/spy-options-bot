import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import traceback
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from technical_analysis.indicators import calculate_indicators
from monitor.health_check import update_status

# === Paths ===
BASE_DIR        = os.path.dirname(__file__)
DATA_PATH       = os.path.join(BASE_DIR, "spy_data.csv")
RAW_MODEL_PATH  = os.path.join(BASE_DIR, "xgb_raw.json")
CAL_MODEL_PATH  = os.path.join(BASE_DIR, "xgb_calibrated.pkl")
LOG_PATH        = os.path.join(BASE_DIR, "retrain_log.csv")
CAL_PLOT_PATH   = os.path.join(BASE_DIR, "calibration_plot.png")

MAX_ROWS = 10000

def send_telegram_message(message: str, photo_path: str = None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
        if photo_path and os.path.exists(photo_path):
            url2 = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                requests.post(url2, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=10)
    except Exception as e:
        logger.error(f"[Telegram Error] {e}")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df.sort_values("timestamp")

def create_labels(df):
    df["future_price"] = df["vwap"].shift(-1)
    df["label"] = (df["future_price"] > df["vwap"]).astype(int)
    return df

def prune_training_data(df):
    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:]
        logger.info(f"[Data Prune] Limited to last {MAX_ROWS} rows")
    return df

def plot_calibration(y_true, prob_pos, filename):
    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated')
    plt.plot([0,1],[0,1], linestyle='--', label='Ideal')
    plt.xlabel('Predicted P')
    plt.ylabel('True Fraction')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

class XGBWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self):
        self.booster = None
        self._fitted = False
        self.feature_names = None

    def fit(self, X, y):
        logger.info("[XGBWrapper] fit() called")
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        dtrain = xgb.DMatrix(X, label=y)
        self.booster = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=100)
        self._fitted = True
        logger.info("[XGBWrapper] Booster trained")
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise ValueError("Wrapper not fitted yet")
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        dmat = xgb.DMatrix(X)
        p = self.booster.predict(dmat)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:,1] > 0.5).astype(int)

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")
        update_status("last_retrain_attempt")

        df = load_data()
        if all(c in df.columns for c in ["open","high","low","close","volume"]):
            df = calculate_indicators(df)
        else:
            logger.info("[Indicators] Skipped — no OHLCV")

        df = df.dropna().reset_index(drop=True)
        if len(df) < 50:
            raise ValueError("Need ≥50 rows to train")

        df = prune_training_data(df)
        df = create_labels(df)

        X = df.drop(columns=["timestamp","future_price","label"])
        y = df["label"]
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

        wrapper = XGBWrapper()
        calibrator = CalibratedClassifierCV(wrapper, method="sigmoid", cv=3)
        calibrator.fit(X_train, y_train)  # wrapper.fit is called internally

        # Save raw model
        wrapper_raw = wrapper
        wrapper_raw.booster.save_model(RAW_MODEL_PATH)

        # Save calibrated
        joblib.dump(calibrator, CAL_MODEL_PATH)

        # Evaluate
        y_pred = calibrator.predict(X_val)
        y_proba = calibrator.predict_proba(X_val)[:,1]
        acc = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_proba)
        plot_calibration(y_val, y_proba, CAL_PLOT_PATH)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH,"a") as f:
            if not exists:
                f.write("timestamp,accuracy,brier\n")
            f.write(f"{now},{acc:.4f},{brier:.4f}\n")

        msg = (
            "📊 ML retrain ✅\n"
            f"Date: {now}\n"
            f"Acc: {acc:.2%}, Brier: {brier:.4f}\n"
            "Cold start with calibration"
        )
        logger.info(msg)
        send_telegram_message(msg, photo_path=CAL_PLOT_PATH)
        update_status("last_retrain")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        send_telegram_message(f"❌ Retrain error: {e}")
        update_status("last_retrain_failed")

if __name__ == "__main__":
    retrain_model()