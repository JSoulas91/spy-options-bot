import os
import pickle
import logging
import xgboost as xgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from ml.build_spy_data_from_meta_log import build_dataset
from monitor.health_check import update_status
from utils.telegram_utils import send_telegram_message
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s",
)
logger = logging.getLogger("retrain")

def load_data():
    df = pd.read_csv("ml/spy_data.csv")
    df.dropna(inplace=True)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    if "label" in df.columns:
        y = df["label"]
        X = df.drop(columns=["label", "pnl"])
    else:
        raise ValueError("Missing label column in spy_data.csv")
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return X, y

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")

        # Build features first
        logger.info("[Preprocess] Running feature builder...")
        build_dataset()
        logger.info("[Preprocess] Feature builder completed successfully.")

        # Load data
        X, y = load_data()

        # Train base XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
        )
        xgb_model.fit(X, y)

        # Save raw booster
        xgb_model.get_booster().save_model("models/xgb_raw.json")
        logger.info("[Save Model] Raw XGBoost booster saved to models/xgb_raw.json")

        # Calibrate
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        # Save calibrated model
        with open("models/xgb_calibrated.pkl", "wb") as f:
            pickle.dump(calibrator, f)
        logger.info("[Save Model] Calibrated model saved to models/xgb_calibrated.pkl")

        # Evaluate
        preds = calibrator.predict(X)
        acc = accuracy_score(y, preds)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        send_telegram_message(f"✅ ML retrained. Accuracy: {acc:.4f}")
        update_status("ml_retrain", "ok")

    except Exception as e:
        logger.critical(f"Fatal error during retraining: {e}")
        try:
            send_telegram_message(f"❌ ML retrain failed: {e}")
        except Exception as tg_error:
            logger.warning(f"❌ Failed to send Telegram message: {tg_error}")
        update_status("ml_retrain", "fail")

if __name__ == "__main__":
    retrain_model()