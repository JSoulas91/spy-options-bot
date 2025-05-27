import joblib
import os

MODEL_PATH = "models/latest_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict(model, features):
    if model is None:
        return 0
    return model.predict([features])[0]

def retrain_model(data, labels):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(data, labels)
    joblib.dump(model, MODEL_PATH)
    return model
