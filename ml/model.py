import joblib
import numpy as np
import os

# Consistent model path (inside ml/models/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'spy_model.pkl')

class SPYModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the ML model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("[Model] Model loaded successfully.")
            except Exception as e:
                print(f"[Model Load Error] {e}")
                self.model = None
        else:
            print("[Model] No model file found. ML predictions will be skipped.")
            self.model = None

    def predict(self, features: np.ndarray):
        """
        Predict confidence score (0.0 to 1.0).
        Returns 0.5 (neutral) if model not loaded or if prediction fails.
        """
        if self.model is None:
            return 0.5
        try:
            prob = self.model.predict_proba([features])[0][1]
            return float(round(prob, 4))  # Return 4-decimal float
        except Exception as e:
            print(f"[Model Prediction Error] {e}")
            return 0.5