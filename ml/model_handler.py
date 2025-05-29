import os
import joblib
import traceback
from sklearn.ensemble import RandomForestClassifier
from utils.logger import bot_logger

# Save directly inside ml/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "spy_model.pkl")

def load_model():
    """Load the trained ML model from disk."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            bot_logger.info("🧠 [Model Handler] Model loaded successfully.")
            return model
        except Exception as e:
            bot_logger.error(f"[Model Load Error] {e}")
            bot_logger.debug(traceback.format_exc())
            return None
    else:
        bot_logger.warning("⚠️ [Model Handler] No model file found.")
        return None

def predict(model, features):
    """Predict using the given model and input features."""
    if model is None:
        bot_logger.warning("[Prediction Warning] Model not loaded. Returning default prediction: 0")
        return 0
    try:
        return model.predict([features])[0]
    except Exception as e:
        bot_logger.error(f"[Prediction Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return 0

def retrain_model(data, labels):
    """Train a new model with the given features and labels."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data, labels)
        joblib.dump(model, MODEL_PATH)

        bot_logger.info("🔁 [Retrain] Model retrained and saved successfully.")
        return model
    except Exception as e:
        bot_logger.error(f"[Retrain Error] {e}")
        bot_logger.debug(traceback.format_exc())
        return None