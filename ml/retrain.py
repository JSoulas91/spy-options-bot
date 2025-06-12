#ml/retrain

import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import traceback
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from technical_analysis.indicators import calculate_indicators

# ─── Health‑check ────────────────────────────────────────────────────────────
from monitor.health_check import update_status

# === Paths ===
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "spy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spy_model.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "retrain_log.csv")

# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
def retrain_model():
    try:
        logger.info("[ML Retraining] Starting retraining process …")
        update_status("last_retrain_attempt")                 # ✅ heartbeat

        # —— Load & prepare data ———————————————————————————
        df = calculate_indicators(load_data())
        df = create_labels(df)

        X = df.drop(columns=["timestamp", "future_close", "label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        model      = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
        model.fit(X_train, y_train)

        # —— Evaluation ————————————————————————————————————
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        acc_pct = f"{acc:.2%}"

        # —— Persist model ————————————————————————————————
        joblib.dump(model, MODEL_PATH)

        # —— Log accuracy ————————————————————————————————
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
        update_status("last_retrain")                          # ✅ heartbeat

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