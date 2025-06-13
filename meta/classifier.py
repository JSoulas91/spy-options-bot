import os
import json
import joblib
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier, Booster, DMatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from config import CLASSIFIER_MODEL_PATH, CLASSIFIER_LOG_PATH, CLASSIFIER_RETRAIN_THRESHOLD

class TradeClassifier:
    def __init__(self):
        self.model = None
        self.calibrator = None
        self.training_data = []
        self.training_labels = []
        self.last_trained = None
        self.load_model()

    def load_model(self):
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            try:
                booster = Booster()
                booster.load_model(CLASSIFIER_MODEL_PATH)
                self.model = XGBClassifier()
                self.model._Booster = booster
                self.model._le = joblib.load(CLASSIFIER_MODEL_PATH.replace('.json', '_le.pkl'))
                self.model.n_classes_ = len(self.model._le.classes_)

                self.calibrator = joblib.load(CLASSIFIER_MODEL_PATH.replace('.json', '_cal.pkl'))
                print("[Classifier] Loaded existing XGBoost model.")
            except Exception as e:
                print(f"[Classifier] Failed to load model: {e}")
        else:
            print("[Classifier] No existing model found. Starting fresh.")

    def save_model(self):
        if self.model is not None:
            self.model.get_booster().save_model(CLASSIFIER_MODEL_PATH)
            joblib.dump(self.model._le, CLASSIFIER_MODEL_PATH.replace('.json', '_le.pkl'))
            if self.calibrator:
                joblib.dump(self.calibrator, CLASSIFIER_MODEL_PATH.replace('.json', '_cal.pkl'))
            print("[Classifier] Model saved.")

    def add_training_sample(self, features: list, label: int):
        self.training_data.append(features)
        self.training_labels.append(label)

        if len(self.training_data) >= CLASSIFIER_RETRAIN_THRESHOLD:
            self.retrain()

    def retrain(self):
        print("[Classifier] Retraining classifier...")
        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        try:
            # Prune old logs to keep size manageable
            if len(X) > 2000:
                X = X[-2000:]
                y = y[-2000:]

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )

            self.model.fit(X_train, y_train)

            self.calibrator = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
            self.calibrator.fit(X_val, y_val)

            self.last_trained = datetime.now().isoformat()
            self.save_model()
            self._log_training(X_val, y_val)
        except Exception as e:
            print(f"[Classifier] Retraining failed: {e}")

    def _log_training(self, X_val, y_val):
        if not os.path.exists(CLASSIFIER_LOG_PATH):
            os.makedirs(CLASSIFIER_LOG_PATH)

        preds = self.calibrator.predict(X_val)
        acc = (preds == y_val).mean()

        log_entry = {
            'timestamp': self.last_trained,
            'validation_accuracy': float(round(acc, 4)),
            'samples_used': len(X_val)
        }

        with open(os.path.join(CLASSIFIER_LOG_PATH, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def predict(self, features: list):
        if self.calibrator is None or self.model is None:
            return None  # Untrained

        try:
            proba = self.calibrator.predict_proba([features])[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])
            return pred_class, confidence
        except Exception as e:
            print(f"[Classifier] Prediction error: {e}")
            return None

    def predict_regime(self, features: list):
        # Optional: predict bull(0), bear(1), or neutral(2) regime
        return self.predict(features)