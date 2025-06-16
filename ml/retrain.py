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
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                requests.post(url_photo, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=10)
    except Exception as e:
        logger.error(f"[Telegram Error] {e}")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    logger.info(f"[Load Data] Loaded {len(df)} rows with semantic columns: {list(df.columns)}")
    return df.sort_values("timestamp")

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["future_price"] = df["vwap"].shift(-1)
    df["label"] = (df["future_price"] > df["vwap"]).astype(int)
    return df

def prune_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:]
        logger.info(f"[Data Prune] Training data pruned to last {MAX_ROWS} rows.")
    return df

def plot_calibration(y_true, prob_pos, filename):
    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=10)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibrated Model')
    plt.plot([0,1],[0,1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Curve)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster=None):
        self.booster = booster
        self._fitted = booster is not None

    def fit(self, X=None, y=None):
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise ValueError("This XGBWrapper instance is not fitted yet.")
        dmatrix = xgb.DMatrix(X)
        probs = self.booster.predict(dmatrix)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

def retrain_model():
    try:
        logger.info("[ML Retraining] Starting forced cold-start XGBoost retraining …")
        update_status("last_retrain_attempt")

        df = load_data()

        required_cols = ["open", "high", "low", "close", "volume"]
        if all(col in df.columns for col in required_cols):
            df = calculate_indicators(df)
        else:
            logger.info("[Indicators] Skipped indicator calculation — OHLCV columns missing.")

        df = df.dropna().reset_index(drop=True)
        if len(df) < 50:
            raise ValueError(f"Not enough data to train. Need at least 50 rows, found {len(df)}.")

        df = prune_training_data(df)
        df = create_labels(df)

        X = df.drop(columns=["timestamp", "future_price", "label"])
        y = df["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.1,
            "max_depth": 4,
            "verbosity": 1,
            "seed": 42,
        }

        # ✅ Force cold-start training
        logger.info("[Force Cold Start] Training new XGBoost model …")
        booster = xgb.train(params, dtrain, num_boost_round=100)
        booster.save_model(RAW_MODEL_PATH)

        wrapper = XGBWrapper(booster)
        wrapper.fit()

        logger.info("[Calibration] Running calibration …")
        calibrator = CalibratedClassifierCV(wrapper, method="sigmoid", cv="prefit")
        calibrator.fit(X_val, y_val)

        y_val_pred = calibrator.predict(X_val)
        y_val_proba = calibrator.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_val_pred)
        brier = brier_score_loss(y_val, y_val_proba)

        plot_calibration(y_val, y_val_proba, CAL_PLOT_PATH)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a") as f:
            if not log_exists:
                f.write("timestamp,accuracy,brier_score\n")
            f.write(f"{now},{acc:.4f},{brier:.4f}\n")

        joblib.dump(calibrator, CAL_MODEL_PATH)

        message = (
            f"📊 *ML Cold Retraining + Calibration Complete*\n"
            f"🗓️  Date: {now.split()[0]}\n"
            f"🎯 Accuracy: *{acc:.2%}*\n"
            f"📉 Brier Score: *{brier:.4f}*\n"
            f"🔥 Warm Start: No (forced cold start)\n"
            f"💾 Models: Raw booster + Calibrated wrapper\n"
            f"✅ Status: Saved & Logged\n"
        )

        logger.info(f"[ML Retraining] Success — Accuracy: {acc:.2%}, Brier: {brier:.4f}")
        send_telegram_message(message, photo_path=CAL_PLOT_PATH)
        update_status("last_retrain")

    except Exception as e:
        logger.critical(f"[ML Retraining] Fatal error: {e}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"❌ *ML Retraining Failed*\n"
            f"🧨 Error: `{str(e)}`\n"
            f"📉 Recovery will be attempted tomorrow."
        )

if __name__ == "__main__":
    retrain_model()