import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'spy_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'retrain_log.csv')

# Initialize Alpaca
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def fetch_data(symbol="SPY", lookback_days=60):
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    bars = alpaca.get_bars(symbol, TimeFrame.Day, start=start_dt.isoformat(), end=end_dt.isoformat()).df

    if bars.empty:
        raise ValueError("No data returned from Alpaca. Market may be closed or unavailable.")
    
    bars = bars.reset_index()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    return bars

def create_labels(df):
    df['future_close'] = df['close'].shift(-1)
    df['label'] = (df['future_close'] > df['close']).astype(int)
    return df.dropna()

def extract_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=5).std()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_10']
    df = df.dropna()
    return df[['return', 'volatility', 'sma_5', 'sma_10', 'sma_ratio', 'label']]

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"[Telegram Error] {e}")

def get_last_accuracy():
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            if not df.empty:
                return df.iloc[-1]['accuracy']
        except Exception:
            return None
    return None

def retrain_model():
    try:
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
        acc_pct = f"{acc:.2%}"

        # Save model and log
        joblib.dump(model, MODEL_PATH)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, 'a') as f:
            if not log_exists:
                f.write("timestamp,accuracy\n")
            f.write(f"{now},{acc:.4f}\n")

        prev_acc = get_last_accuracy()
        comparison = f"Compared to last: *{float(prev_acc):.2%}*" if prev_acc else "First run"

        message = (
            f"📊 *ML Retraining Complete*\n"
            f"🗓️  Date: {now.split()[0]}\n"
            f"🎯 Accuracy: *{acc_pct}*\n"
            f"📈 {comparison}\n"
            f"💾 Model: `spy_model.pkl`\n"
            f"📈 Status: ✅ Saved & Logged"
        )

        print(message)
        send_telegram_message(message)

    except Exception as e:
        error_msg = f"❌ *ML Retraining Failed*\nReason: `{str(e)}`"
        print(error_msg)
        send_telegram_message(error_msg)

if __name__ == "__main__":
    retrain_model()