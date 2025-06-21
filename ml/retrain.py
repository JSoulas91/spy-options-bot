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

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv("ml/spy_data.csv")
    logger.info(f"[Load Data] Raw: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop(columns=["timestamp"], errors="ignore")

    null_counts = df.isnull().sum()
    missing_cols = null_counts[null_counts > 0]
    if not missing_cols.empty:
        logger.info(f"[Load Data] Missing values before cleanup:\n{missing_cols}")

    df = df[df["label"].notnull()]
    if "pnl" in df.columns:
        df = df[df["pnl"].notnull()]

    df = df.fillna(method="ffill").fillna(method="bfill")

    logger.info(f"[Load Data] After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def timestamped_path(base: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(MODEL_DIR, f"{base}_{ts}.{ext}")

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")

        logger.info("[Preprocess] Running feature builder...")
        build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")

        df = load_data()
        logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")

        y = df["label"]
        drop_cols = ["label", "pnl", "trade_success_prob", "predicted_direction", "classifier_entropy"]
        X = df.drop(columns=drop_cols, errors="ignore")

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

        raw_model_path = os.path.join(MODEL_DIR, "xgb_raw.json")
        xgb_model.get_booster().save_model(raw_model_path)
        logger.info(f"[Save Model] Raw XGBoost booster saved: {raw_model_path}")

        backup_raw = timestamped_path("xgb_raw", "json")
        xgb_model.get_booster().save_model(backup_raw)
        logger.info(f"[Backup Model] Timestamped backup: {backup_raw}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        calibrated_path = os.path.join(MODEL_DIR, "xgb_calibrated.pkl")
        joblib.dump(calibrator, calibrated_path)
        logger.info(f"[Save Model] Calibrated model saved: {calibrated_path}")

        backup_cal = timestamped_path("xgb_calibrated", "pkl")
        joblib.dump(calibrator, backup_cal)
        logger.info(f"[Backup Model] Timestamped backup: {backup_cal}")

        acc = calibrator.score(X, y)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        # Accuracy log
        with open("ml/accuracy_log.txt", "a") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}, accuracy={acc:.4f}, rows={len(df)}\n")

        update_status("last_retrain", "ok")
        send_telegram_message(f"✅ ML retrained successfully\n📊 Accuracy: {acc:.4f}\n📁 Rows: {len(df)}")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}")
        update_status("last_retrain", "fail")
        try:
            send_telegram_message(f"❌ ML retrain failed:\n{e}")
        except Exception as inner:
            logger.warning(f"[Telegram Fail] {inner}")

if __name__ == "__main__":
    retrain_model()