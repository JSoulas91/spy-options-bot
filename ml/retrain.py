import os
import sys
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils.logger import bot_logger as logger
from technical_analysis.indicators import calculate_indicators
from monitor.health_check import update_status
from utils.telegram_utils import send_telegram_message

from ml.build_spy_data_from_meta_log import build_dataset

DATA_PATH = "ml/spy_data.csv"
MODEL_PATH = "ml/xgb_raw.json"
BACKUP_DIR = "models/"
ACCURACY_LOG = "ml/accuracy_log.txt"

def run_build_dataset():
    logger.info("[Preprocess] Running feature builder...")
    try:
        build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")
    except Exception as e:
        logger.error(f"[Preprocess Error] {e}")
        raise

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

    drop_cols = ['timestamp', 'label', 'pnl'] if 'label' in df.columns else ['timestamp', 'pnl']
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df['label']
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(X.columns)}")
    return X, y, df

def retrain_model():
    logger.info("[ML Retraining] Started")
    run_build_dataset()
    X, y, df = load_data()

    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9, use_label_encoder=False, eval_metric='logloss')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        calibrator = CalibratedClassifierCV(estimator=model, method='isotonic', cv=skf)
        calibrator.fit(X, y)
        preds = calibrator.predict(X)
        acc = accuracy_score(y, preds)

        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        # Save raw model
        calibrator.base_estimator.save_model(MODEL_PATH)

        # Save backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"xgb_model_{timestamp}.json")
        os.makedirs(BACKUP_DIR, exist_ok=True)
        calibrator.base_estimator.save_model(backup_path)

        # Log accuracy
        with open(ACCURACY_LOG, "a") as f:
            f.write(f"{timestamp},{acc:.4f}\n")

        # Telegram alert (HTML formatted)
        send_telegram_message(
            f"✅ <b>Classifier Retrained</b>\n"
            f"<b>Samples:</b> {len(df)}\n"
            f"<b>Accuracy:</b> {acc:.4f}"
        )

        update_status("ml_retrain", "success")
    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}", exc_info=True)
        send_telegram_message(
            f"❌ <b>Retraining failed</b>\n<pre>{str(e)}</pre>"
        )
        update_status("ml_retrain", "fail")

if __name__ == "__main__":
    retrain_model()