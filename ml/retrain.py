import os
import logging
from datetime import datetime
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split

# Config paths
RAW_MODEL_PATH = "models/xgb_raw.json"
CAL_MODEL_PATH = "models/xgb_calibrated.pkl"
CAL_PLOT_PATH = "reports/calibration_plot.png"
LOG_PATH = "logs/retrain_log.csv"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s",
)
logger = logging.getLogger("retrain")

def send_telegram_message(text, photo_path=None):
    # Placeholder: implement your Telegram bot send logic here
    pass

def update_status(key):
    # Placeholder: implement your heartbeat/status update logic here
    pass

def load_data():
    path = "ml/spy_data.csv"
    df = pd.read_csv(path)
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df

def prune_training_data(df, max_rows=10000):
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
        logger.info(f"[Data Prune] Limited to last {max_rows} rows")
    return df

def calculate_indicators(df):
    # If you have OHLCV data, calculate indicators here
    # Example: placeholder
    return df

def create_labels(df):
    df['label'] = df['label'].astype(int)
    return df

def plot_calibration(y_true, y_prob, save_path):
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Calibration curve")
    plt.plot([0,1], [0,1], linestyle='--', label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"[Plot] Saved calibration curve to {save_path}")

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

        X = df.drop(columns=["timestamp","label"])
        y = df["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=100,
            random_state=42,
            # use_label_encoder deprecated, removed
        )

        calibrator = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrator.fit(X_train, y_train)

        # Save raw model (uncalibrated) from one fold of calibrated classifiers
        try:
            fitted_model = calibrator.calibrated_classifiers_[0][0]
            if isinstance(fitted_model, xgb.XGBClassifier):
                fitted_model.get_booster().save_model(RAW_MODEL_PATH)
                logger.info(f"[Model] Saved raw XGB model to {RAW_MODEL_PATH}")
            else:
                logger.warning("[Model] Calibrated base is not an XGBClassifier — skipping raw model save")
        except Exception as e:
            logger.warning(f"[Model] Could not save raw model: {e}")

        # Save calibrated model
        joblib.dump(calibrator, CAL_MODEL_PATH)
        logger.info(f"[Model] Saved calibrated model to {CAL_MODEL_PATH}")

        y_pred = calibrator.predict(X_val)
        y_proba = calibrator.predict_proba(X_val)[:,1]
        acc = accuracy_score(y_val, y_pred)
        brier = brier_score_loss(y_val, y_proba)

        plot_calibration(y_val, y_proba, CAL_PLOT_PATH)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a") as f:
            if not exists:
                f.write("timestamp,accuracy,brier\n")
            f.write(f"{now},{acc:.4f},{brier:.4f}\n")

        msg = (
            f"📊 ML retrain ✅\n"
            f"Date: {now}\n"
            f"Accuracy: {acc:.2%}\n"
            f"Brier Score: {brier:.4f}\n"
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