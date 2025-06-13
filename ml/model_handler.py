# ml/model_handler.py
import os
import xgboost as xgb
import traceback
import numpy as np
from utils.logger import bot_logger

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_raw.json")

def load_model():
    """Load the trained XGBoost model from disk."""
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            bot_logger.info("🧠 [Model Handler] XGBoost model loaded successfully.")
            return model
        except Exception as e:
            bot_logger.error(f"[Model Load Error] {e}")
            bot_logger.debug(traceback.format_exc())
            return None
    else:
        bot_logger.warning("⚠️ [Model Handler] No model file found.")
        return None

def predict(model, features: np.ndarray):
    """Predict using the loaded XGBoost model."""
    if model is None:
        bot_logger.warning("[Prediction Warning] Model not loaded. Returning default prediction: 0.5")
        return 0.5
    try:
        prob = model.predict_proba(np.array([features]))[0][1]
        return float(round(prob, 4))
    except Exception as e:
        bot_logger.error(f"[Prediction Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return 0.5