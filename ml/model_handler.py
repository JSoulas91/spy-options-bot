import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Use consistent and OS-safe path
MODEL_DIR = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.pkl")

# === Load Trained Model ===
def load_model():
    """Load the trained ML model from disk."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[Model Load Error] {e}")
            return None
    return None

# === Make Prediction ===
def predict(model, features):
    """Predict using the given model and input features."""
    if model is None:
        print("[Prediction Warning] Model not loaded.")
        return 0
    try:
        return model.predict([features])[0]
    except Exception as e:
        print(f"[Prediction Error] {e}")
        return 0

# === Retrain Model with New Data ===
def retrain_model(data, labels):
    """Train a new model with the given features and labels."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data, labels)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("[Model Retrained] New model saved successfully.")
        return model
    except Exception as e:
        print(f"[Retrain Error] {e}")
        return None