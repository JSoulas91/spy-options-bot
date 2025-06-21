import os
import pandas as pd
import logging
import joblib
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    df = df.drop(columns=["timestamp"], errors="ignore")
    df = df[df["label"].notnull()].fillna(method="ffill").fillna(method="bfill")
    return df

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")

        build_dataset()

        df = load_data()
        y = df["label"]
        X = df.drop(columns=["label", "pnl"], errors="ignore")

        pos_weight = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        xgb_model.fit(X, y)

        os.makedirs("models", exist_ok=True)
        model_path = f"models/xgb_raw.json"
        xgb_model.get_booster().save_model(model_path)
        logger.info(f"[Save Model] Raw booster saved to {model_path}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=skf)
        calibrator.fit(X, y)

        joblib.dump(calibrator, "models/xgb_calibrated.pkl")
        logger.info("[Save Model] Calibrated model saved.")

        acc = calibrator.score(X, y)
        logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

        feature_importance = xgb_model.feature_importances_
        top_feats = sorted(zip(X.columns, feature_importance), key=lambda x: -x[1])
        logger.info(f"[Top Features]\n" + "\n".join(f"{f:<20s}: {w:.4f}" for f, w in top_feats[:10]))

        update_status("last_retrain", "ok")
        send_telegram_message(f"✅ ML retrained\nAccuracy: {acc:.4f}")

    except Exception as e:
        logger.critical(f"[Retrain] Fatal error: {e}")
        update_status("last_retrain", "fail")
        try:
            send_telegram_message(f"❌ ML retrain failed:\n{e}")
        except Exception as inner:
            logger.warning(f"[Telegram Fail] {inner}")

if __name__ == "__main__":
    retrain_model()