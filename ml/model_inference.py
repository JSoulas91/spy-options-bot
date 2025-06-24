import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CAL_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models/xgb_calibrated.pkl"))

class ModelInference:
    def __init__(self):
        if not os.path.exists(CAL_MODEL_PATH):
            raise FileNotFoundError(f"Calibrated model not found at {CAL_MODEL_PATH}. Run retrain first.")
        self.model = joblib.load(CAL_MODEL_PATH)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns probability of positive class (success).
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_confidence(self, X: pd.DataFrame) -> dict:
        """
        Returns:
        - predicted class (0 or 1)
        - probability of positive class
        - entropy of class probabilities (for uncertainty)
        - full probability distribution over classes
        """
        probs = self.model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[1])  # assuming binary classification [P(class=0), P(class=1)]
        entropy = -np.sum(probs * np.log(probs + 1e-9))  # entropy of prediction

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "entropy": entropy,
            "probs": probs.tolist()
        }