import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spy_model.pkl')

class SPYModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None
            print("[Model] No model file found. ML predictions will be skipped.")

    def predict(self, features: np.ndarray):
        if self.model is None:
            return 0.5  # Neutral confidence if no model
        try:
            prob = self.model.predict_proba([features])[0][1]
            return prob
        except Exception as e:
            print(f"[Model] Prediction error: {e}")
            return 0.5
