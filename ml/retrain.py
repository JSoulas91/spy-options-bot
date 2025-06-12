import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import traceback
import requests
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from technical_analysis.indicators import calculate_indicators
from monitor.health_check import update_status

# === Paths ===
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "spy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spy_model.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "retrain_log.csv")

# === Config ===
MAX_ROWS = 10000  # ⛏️ Keep only last N rows of training data

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").dropna().reset_index(drop=True)

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["future_close"] = df["close"].shift(-1)
    df["label"] = (df["future_close"] > df["close"]).astype(int)
    return df.dropna()

def send_telegram_message(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logger.error(f"[Telegram Error] {e}")

def get_last_accuracy():
    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            if not df.empty:
                return df.iloc[-1]["accuracy"]
    except Exception as e:
        logger.warning(f"[ML Retraining] Couldn't load previous accuracy: {e}")
    return None

def prune_training_data(df: pd.DataFrame):
    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:]
        logger.info(f"[Data Prune] Training data pruned to last {MAX_ROWS} rows.")
    return df

def retrain_model():
    try:
        logger.info("[ML Retraining] Starting XGBoost retraining process …")
        update_status("last_retrain_attempt")

        df = calculate_indicators(load_data())
        df = create_labels(df)
        df = prune_training_data(df)

        X = df.drop(columns=["timestamp", "future_close", "label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        base_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_pct = f"{acc:.2%}"

        joblib.dump(model, MODEL_PATH)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a") as f:
            if not log_exists:
                f.write("timestamp,accuracy\n")
            f.write(f"{now},{acc:.4f}\n")

        prev_acc = get_last_accuracy()
        comparison = (
            f"Compared to last: *{float(prev_acc):.2%}*"
            if prev_acc else "First run or previous accuracy unavailable"
        )

        message = (
            f"📊 *ML Retraining Complete*\n"
            f"🗓️  Date: {now.split()[0]}\n"
            f"🎯 Accuracy: *{acc_pct}*\n"
            f"📈 {comparison}\n"
            f"💾 Model: `spy_model.pkl`\n"
            f"✅ Status: Saved & Logged"
        )

        logger.info(f"[ML Retraining] Success — Accuracy: {acc_pct}")
        send_telegram_message(message)
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