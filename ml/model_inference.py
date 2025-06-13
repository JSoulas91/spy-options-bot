import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR       = os.path.dirname(__file__)
CAL_MODEL_PATH = os.path.join(BASE_DIR, "xgb_calibrated.pkl")

class ModelInference:
    def __init__(self):
        if not os.path.exists(CAL_MODEL_PATH):
            raise FileNotFoundError("Calibrated model not found. Run retrain first.")
        self.model = joblib.load(CAL_MODEL_PATH)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Return calibrated probabilities for class 1
        return self.model.predict_proba(X)[:,1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Binary predictions
        return self.model.predict(X)

# Example usage:
# mi = ModelInference()
# probs = mi.predict_proba(new_features_df)
# preds = mi.predict(new_features_df)
