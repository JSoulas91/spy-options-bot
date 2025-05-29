import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
import traceback

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)
from utils.logger import bot_logger as logger

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'spy_model.pkl')
LOG_PATH = os.path.join(BASE_DIR, 'retrain_log.csv')

# === Alpaca API Client ===
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def fetch_data(symbol="SPY", lookback_days=60):
    try:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback_days)
        bars = alpaca.get_bars(symbol, TimeFrame.Day, start=start_dt.isoformat(), end=end_dt.isoformat()).df

        if bars.empty:
            raise ValueError("No data returned from Alpaca. Market may be closed or unavailable.")
        
        bars = bars.reset_index()
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        return bars

    except Exception as e:
        logger.error(f"[ML Retraining] Error in fetch_data: {e}")
        raise

def create_labels(df):
    try:
        df['future_close'] = df['close'].shift(-1)
        df['label'] = (df['future_close'] > df['close']).astype(int)
        return df.dropna()
    except Exception as e:
        logger.error(f"[ML Retraining] Error in create_labels: {e}")
        raise

def extract_features(df):
    try:
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=5).std()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_ratio'] = df['sma_5'] / df['sma_10']
        df = df.dropna()
        return df[['return', 'volatility', 'sma_5', 'sma_10', 'sma_ratio', 'label']]
    except Exception as e:
        logger.error(f"[ML Retraining] Error in extract_features: {e}")
        raise

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logger.error(f"[Telegram Error] {e}")

def get_last_accuracy():
    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            if not df.empty:
                return df.iloc[-1]['accuracy']
    except Exception as e:
        logger.warning(f"[ML Retraining] Couldn't load previous accuracy: {e}")
    return None

def retrain_model():
    try:
        logger.info("[ML Retraining] Starting retraining process...")
        
        # === Load + Preprocess ===
        raw = fetch_data()
        labeled = create_labels(raw)
        data = extract_features(labeled)

        # === Train/Test Split ===
        X = data.drop(columns=['label'])
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # === Train Model ===
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_pct = f"{acc:.2%}"

        # === Save Model ===
        joblib.dump(model, MODEL_PATH)

        # === Log Accuracy ===
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_exists = os.path.exists(LOG_PATH)
        try:
            with open(LOG_PATH, 'a') as f:
                if not log_exists:
                    f.write("timestamp,accuracy\n")
                f.write(f"{now},{acc:.4f}\n")
        except Exception as e:
            logger.error(f"[ML Retraining] Failed to write to retrain_log.csv: {e}")

        # === Accuracy Comparison ===
        prev_acc = get_last_accuracy()
        if prev_acc:
            comparison = f"Compared to last: *{float(prev_acc):.2%}*"
        else:
            comparison = "First run or previous accuracy unavailable"

        # === Notify ===
        message = (
            f"📊 *ML Retraining Complete*\n"
            f"🗓️  Date: {now.split()[0]}\n"
            f"🎯 Accuracy: *{acc_pct}*\n"
            f"📈 {comparison}\n"
            f"💾 Model: `spy_model.pkl`\n"
            f"📈 Status: ✅ Saved & Logged"
        )

        logger.info(f"[ML Retraining] Success — Accuracy: {acc_pct}")
        logger.info(message.replace("*", "").replace("`", ""))  # plain log version
        send_telegram_message(message)

    except Exception as e:
        logger.critical(f"[ML Retraining] Fatal error during model retraining: {e}")
        logger.debug(traceback.format_exc())
        send_telegram_message(
            f"❌ *ML Retraining Failed*\n"
            f"🧨 Error: `{str(e)}`\n"
            f"📉 Recovery will be attempted tomorrow."
        )

if __name__ == "__main__":
    retrain_model()