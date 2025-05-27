import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_BASE_URL
from model_handler import extract_features  # Reuse your existing feature extractor

# === Setup Alpaca API ===
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_BASE_URL)

def fetch_data(symbol="SPY", lookback_days=60):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_days)
    bars = alpaca.get_bars(symbol, TimeFrame.Day, start=start_dt.isoformat(), end=end_dt.isoformat()).df
    bars = bars.reset_index()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    return bars

def create_labels(df):
    df['future_close'] = df['close'].shift(-1)
    df['label'] = (df['future_close'] > df['close']).astype(int)
    df = df.dropna()
    return df

def retrain_model():
    print("[Retraining ML Model] Fetching data...")
    raw_data = fetch_data()
    raw_data = create_labels(raw_data)

    print("[Retraining] Extracting features...")
    features_df = extract_features(raw_data)
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    print("[Retraining] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[Retraining] Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Retraining] Accuracy on test set: {acc:.2f}")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model, model_path)
    print(f"[Retraining] Model saved to {model_path}")

if __name__ == "__main__":
    retrain_model()