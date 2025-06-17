import os
import pandas as pd
import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# === Paths ===
DATA_PATH = "ml/spy_data.csv"
RAW_MODEL_PATH = "models/xgb_raw.json"
CAL_MODEL_PATH = "models/xgb_calibrated.pkl"
ACCURACY_LOG_PATH = "ml/accuracy_log.txt"
CALIBRATION_PLOT_PATH = "ml/calibration_plot.png"

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(funcName)s — Line %(lineno)d — %(message)s",
)
logger = logging.getLogger("retrain")

# === Load training data ===
def load_data():
    df = pd.read_csv(DATA_PATH)
    logger.info(f"[Load Data] Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df

# === Prune data to limit memory usage ===
def prune_training_data(df, max_rows=10000):
    if len(df) > max_rows:
        df = df.iloc[-max_rows:]
        logger.info(f"[Data Prune] Limited to last {max_rows} rows")
    return df

# === Main retraining logic ===
def retrain_model():
    logger.info("[ML Retraining] Started")
    df = load_data()

    # Drop unused columns
    drop_cols = ["timestamp", "pnl"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    if "open" not in df.columns or "high" not in df.columns:
        logger.info("[Indicators] Skipped — no OHLCV")

    df = prune_training_data(df)

    # Separate features and target
    X = df.drop(columns=["label"])
    y = df["label"]

    # Initialize XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    # Cross-validated calibrated classifier
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv=StratifiedKFold(n_splits=5))

    try:
        calibrator.fit(X, y)
    except Exception as e:
        logger.critical(f"[Training Error] {e}")
        return

    y_pred = calibrator.predict(X)
    acc = accuracy_score(y, y_pred)
    logger.info(f"[Accuracy] In-sample accuracy: {acc:.4f}")

    # Log accuracy
    with open(ACCURACY_LOG_PATH, "a") as f:
        f.write(f"{acc:.4f}\n")

    # Save calibration plot
    prob_pos = calibrator.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="XGBoost (isotonic)")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(CALIBRATION_PLOT_PATH)
    plt.close()

    # === Save models ===
    try:
        # Ensure model dir exists
        os.makedirs(os.path.dirname(RAW_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CAL_MODEL_PATH), exist_ok=True)

        # Save underlying raw XGBoost model
        fitted_model = calibrator.base_estimator
        if isinstance(fitted_model, list):
            fitted_model = fitted_model[0]
        fitted_model.save_model(RAW_MODEL_PATH)
        logger.info(f"[Model Saved] Raw XGBoost model saved to {RAW_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"[Model] Could not save raw model: {e}")

    try:
        joblib.dump(calibrator, CAL_MODEL_PATH)
        logger.info(f"[Model Saved] Calibrated model saved to {CAL_MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")

if __name__ == "__main__":
    retrain_model()