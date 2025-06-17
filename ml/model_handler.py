import os
import joblib
import pandas as pd
from typing import Dict

from ml.constants import MODEL_PATH, BASE_DIR

CALIBRATED_MODEL_PATH = os.path.join(BASE_DIR, "xgb_calibrated.pkl")

def load_model(path: str = CALIBRATED_MODEL_PATH):
    """
    Load the calibrated XGBoost classifier model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    return model

def predict(input_features: Dict[str, float]) -> float:
    """
    Predict the probability of a successful trade using the calibrated model.
    """
    model = load_model()
    X = pd.DataFrame([input_features])

    # Ensure all expected features are present in input
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required input features: {missing}")
        X = X[model.feature_names_in_]

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba — expected a calibrated classifier.")

    y_prob = model.predict_proba(X)[0, 1]
    return float(y_prob)

def predict_label(input_features: Dict[str, float]) -> int:
    """
    Predict a binary label (0 or 1) based on the model.
    """
    model = load_model()
    X = pd.DataFrame([input_features])

    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required input features: {missing}")
        X = X[model.feature_names_in_]

    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not support predict — expected a classifier.")
    
    y_pred = model.predict(X)[0]
    return int(y_pred)