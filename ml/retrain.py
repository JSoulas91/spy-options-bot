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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain")

FEATURE_NAMES = [
    "confidence",
    "setup_quality",
    "vix",
    "realized_vol",
    "trade_type",
    "total_signals_today",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "vwap",
    "atr_14",
    "adx_14"
]

def load_data():
    df = pd.read_csv("ml/spy_data.csv")
    df = df.drop(columns=["timestamp"], errors="ignore")
    df = df[df["label"].notnull()].ffill().bfill()

    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        logger.warning(f"[Data Load] Missing columns: {missing_cols}")
        raise ValueError(f"Missing required features: {missing_cols}")

    return df

def retrain_model():
    try:
        logger.info("[ML Retraining] Started")
        build_dataset()

        df = load_data()
        y = df["label"]
        X = df[FEATURE_NAMES]

        num_pos = sum(y)
        num_neg = len(y) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        if num_pos < 50 or num_neg < 50:
            send_telegram_message(f"⚠️ Imbalanced dataset: pos={num_pos}, neg={num_neg}")

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = f"models/xgb_raw_{timestamp}.json"
        xgb_model.get_booster().save_model(raw_path)
        logger.info(f"[Save Model] Raw saved to {raw_path}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calibrator = CalibratedClassifierCV(xgb_model, method="isotonic", cv=skf)
        calibrator.fit(X, y)
        calibrated_path = f"models/xgb_calibrated_{timestamp}.pkl"
        joblib.dump(calibrator, calibrated_path)
        logger.info("[Save Model] Calibrated model saved")
        # Save a pointer to the latest model path
        with open("models/latest_model_path.txt", "w") as f:
            f.write(calibrated_path)
        logger.info(f"[Save Model] Latest model path recorded to models/latest_model_path.txt")
        
        # Accuracy
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        acc_in = calibrator.score(X_train, y_train)
        acc_val = calibrator.score(X_val, y_val)
        logger.info(f"[Accuracy] In-sample: {acc_in:.4f}, Out-of-sample: {acc_val:.4f}")

        with open("models/accuracy_log.txt", "a") as f:
            f.write(f"{timestamp},in={acc_in:.4f},val={acc_val:.4f},pos={num_pos},neg={num_neg}\n")

        # Top features
        importances = xgb_model.feature_importances_
        top_feats = sorted(zip(X.columns, importances), key=lambda x: -x[1])
        logger.info("[Top Features]\n" + "\n".join(f"{f:<20}: {w:.4f}" for f, w in top_feats[:10]))

        send_telegram_message(
            f"✅ ML retrained\nIn-sample acc: {acc_in:.4f}\nOOS acc: {acc_val:.4f}\nModel: {calibrated_path}"
        )
        update_status("last_retrain", "ok")

    except Exception as e:
        logger.exception("[Retrain] Fatal error")
        update_status("last_retrain", "fail")
        try:
            send_telegram_message(f"❌ ML retrain failed:\n{e}")
        except Exception:
            pass

if __name__ == "__main__":
    retrain_model()