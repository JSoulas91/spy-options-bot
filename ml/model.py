import os
import joblib
import numpy as np
import traceback
from utils.logger import bot_logger

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spy_model.pkl')

class SPYModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the ML model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                bot_logger.info("🧠 [Model] Model loaded successfully.")
            except Exception as e:
                bot_logger.error(f"[Model Load Error] {e}")
                bot_logger.debug(traceback.format_exc())
                self.model = None
        else:
            bot_logger.warning("⚠️ [Model] No model file found. ML predictions will be skipped.")
            self.model = None

    def predict(self, features: np.ndarray):
        """Predict confidence score (0.0 to 1.0). Return 0.5 if model unavailable or error."""
        if self.model is None:
            bot_logger.warning("[Model] No model loaded. Returning neutral confidence (0.5).")
            return 0.5
        try:
            prob = self.model.predict_proba([features])[0][1]
            return float(round(prob, 4))
        except Exception as e:
            bot_logger.error(f"[Model Prediction Error] {e}")
            bot_logger.debug(traceback.format_exc())
            return 0.5