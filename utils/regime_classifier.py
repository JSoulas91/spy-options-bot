import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

MODEL_PATH = "utils/regime_kmeans_model.pkl"

def prepare_features(spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for clustering from SPY and VIX data.

    Returns a dataframe of features aligned by date.
    """
    df = spy_data[['close']].rename(columns={'close': 'spy_close'}).copy()
    df['vix_close'] = vix_data['close']
    df.dropna(inplace=True)

    # Features:
    # 1) SPY 10-day return
    df['spy_return_10d'] = df['spy_close'].pct_change(10)
    # 2) VIX 5-day mean
    df['vix_mean_5d'] = df['vix_close'].rolling(5).mean()
    # 3) VIX 10-day mean
    df['vix_mean_10d'] = df['vix_close'].rolling(10).mean()
    # 4) VIX short-term trend (5d mean - 10d mean)
    df['vix_trend'] = df['vix_mean_5d'] - df['vix_mean_10d']

    df.dropna(inplace=True)
    features = df[['spy_return_10d', 'vix_mean_5d', 'vix_mean_10d', 'vix_trend']]
    return features

def train_clustering_model(spy_data: pd.DataFrame, vix_data: pd.DataFrame, n_clusters=3) -> KMeans:
    """
    Train a KMeans clustering model on market features.

    Saves the trained model to disk for later use.
    """
    features = prepare_features(spy_data, vix_data)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(features)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Clustering model trained and saved to {MODEL_PATH}")
    return model

def load_clustering_model() -> KMeans:
    """
    Load the saved KMeans clustering model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def classify_regime(spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
    """
    Classify current market regime using the clustering model.

    Returns a regime label string like "regime_0", "regime_1", etc.

    You can map these to meaningful names after inspecting cluster centers.
    """
    model = load_clustering_model()
    features = prepare_features(spy_data, vix_data)
    latest_features = features.iloc[[-1]]  # last row as 2D dataframe

    cluster_idx = model.predict(latest_features)[0]

    # Map cluster indices to regime names (you can customize this mapping)
    regime_names = {
        0: "regime_0",
        1: "regime_1",
        2: "regime_2"
    }
    return regime_names.get(cluster_idx, f"regime_{cluster_idx}")

if __name__ == "__main__":
    # Example usage:
    # Load your historical SPY and VIX data here, e.g. from CSVs
    spy_data = pd.read_csv("data/spy.csv", parse_dates=['date'], index_col='date')
    vix_data = pd.read_csv("data/vix.csv", parse_dates=['date'], index_col='date')

    # Train model (only once or periodically)
    train_clustering_model(spy_data, vix_data, n_clusters=3)

    # Classify current regime
    current_regime = classify_regime(spy_data, vix_data)
    print(f"Current market regime: {current_regime}")