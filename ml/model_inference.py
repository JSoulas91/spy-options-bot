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
        
    def wrap_classifier_output(raw_output: dict) -> dict:
        """
        Converts binary classifier output into meta-state compatible multi-class format.
    
        Args:
            raw_output (dict): Output from predict_with_confidence(), must include:
                - "confidence": float
                - "predicted_class": int (0 or 1)
                - "entropy": float
                - "probs": list of [P(class=0), P(class=1)]
    
        Returns:
            dict: Formatted output for meta-state functions:
                - trade_success_prob: float
                - predicted_direction: int (0=down, 1=up, 2=flat)
                - class_probabilities: list of 3 floats (down, up, flat)
                - entropy: float
        """
        confidence = raw_output.get("confidence", 0.5)
        predicted_class = raw_output.get("predicted_class", -1)
        entropy = raw_output.get("entropy", 1.0)
        probs = raw_output.get("probs", [0.5, 0.5])
    
        # Heuristic: use confidence to determine up/down direction
        predicted_direction = 1 if confidence > 0.55 else 0  # up if confident
    
        # Extend binary probs to 3-class format [down, up, flat]
        if isinstance(probs, list) and len(probs) == 2:
            class_probabilities = [probs[0], probs[1], 0.0]  # no flat class in binary
        else:
            class_probabilities = [0.5, 0.5, 0.0]  # fallback
    
        return {
            "trade_success_prob": float(confidence),
            "predicted_direction": predicted_direction,
            "class_probabilities": class_probabilities,
            "entropy": float(entropy)
        }