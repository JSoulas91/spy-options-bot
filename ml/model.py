# ml/model.py
import os
import numpy as np
import traceback
import xgboost as xgb
from utils.logger import bot_logger

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgb_raw.json')

class SPYModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load XGBoost model from disk (raw JSON format)."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(MODEL_PATH)
                bot_logger.info("🧠 [Model] XGBoost model loaded successfully.")
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
            prob = self.model.predict_proba(np.array([features]))[0][1]
            return float(round(prob, 4))
        except Exception as e:
            bot_logger.error(f"[Model Prediction Error] {e}")
            bot_logger.debug(traceback.format_exc())
            return 0.5