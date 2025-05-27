import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_BASE_URL

# === Model Save Path ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'spy_model.pkl')

# === Initialize Alpaca ===
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_BASE_URL)

def fetch_data(symbol="SPY", lookback_days=60):
    """Fetch historical OHLCV data from Alpaca."""
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    bars = alpaca.get_bars(symbol, TimeFrame.Day, start=start_dt.isoformat(), end=end_dt.isoformat()).df
    bars = bars.reset_index()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    return bars

def create_labels(df):
    """Create binary labels based on whether the next day's close is higher."""
    df['future_close'] = df['close'].shift(-1)
    df['label'] = (df['future_close'] > df['close']).astype(int)
    return df.dropna()

def extract_features(df):
    """Feature engineering — simple version."""
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=5).std()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_10']
    df = df.dropna()
    return df[['return', 'volatility', 'sma_5', 'sma_10', 'sma_ratio', 'label']]

def retrain_model():
    print("[ML Retraining] Fetching data...")
    raw = fetch_data()
    labeled = create_labels(raw)
    data = extract_features(labeled)

    X = data.drop(columns=['label'])
    y = data['label']

    print("[ML Retraining] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[ML Retraining] Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ML Retraining] Accuracy: {acc:.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[ML Retraining] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    retrain_model()