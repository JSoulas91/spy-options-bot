import os
import pandas as pd
import logging
import joblib
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from utils.telegram_utils import send_telegram_message
from monitor.health_check import update_status
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from ml.build_spy_data_from_meta_log import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s"
)
logger = logging.getLogger("retrain")

def load_data():
    df = pd.read_csv("ml/spy_data.csv")
    logger.info(f"[Load Data] Raw: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop(columns=["timestamp"], errors="ignore")

    null_counts = df.isnull().sum()
    missing_cols = null_counts[null_counts > 0]
    if not missing_cols.empty:
        logger.info(f"[Load Data] Missing values before cleanup:\n{missing_cols}")

    # Drop only rows with missing labels (and pnl if present)
    df = df[df["label"].notnull()]
    if "pnl" in df.columns:
        df = df[df["pnl"].notnull()]

    # Fill missing values in remaining columns
    df = df.fillna(method="ffill").fillna(method="bfill")

    logger.info(f"[Load Data] After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")

        logger.info("[Preprocess] Running feature builder...")
        build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")

        df = load_data()
        logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")

        y = df["label"]
        X = df.drop(columns=["label", "pnl"], errors="ignore")

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        xgb_model.fit(X, y)

        os.makedirs("models", exist_ok=True)
        xgb_model.get_booster().save_model("models/xgb_raw.json")
        logger.info("[Save Model] Raw XGBoost booster saved to models/xgb_raw.json")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        joblib.dump(calibrator, "models/xgb_calibrated.pkl")
        logger.info("[Save Model] Calibrated model saved to models/xgb_calibrated.pkl")

        acc = calibrator.score(X, y)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        update_status("last_retrain", "ok")
        send_telegram_message(f"✅ ML retrained successfully.\nAccuracy: {acc:.4f}")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}")
        update_status("last_retrain", "fail")
        try:
            send_telegram_message(f"❌ ML retrain failed:\n{e}")
        except Exception as inner:
            logger.warning(f"❌ Failed to send Telegram message: {inner}")

if __name__ == "__main__":
    retrain_model()