import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR       = os.path.dirname(__file__)
CAL_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models/xgb_calibrated.pkl"))

class ModelInference:
    def __init__(self):
        if not os.path.exists(CAL_MODEL_PATH):
            raise FileNotFoundError(f"Calibrated model not found at {CAL_MODEL_PATH}. Run retrain first.")
        self.model = joblib.load(CAL_MODEL_PATH)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)